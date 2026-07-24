## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 7200 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.85 + 99.00 = 101.85 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -70.3169252, upper bound: 70.3169252

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1912455, upper bound: 70.1912455
time: 225.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1912455, upper bound: 70.1912455
time: 225.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 450.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 450.74
Output dim: 4, lower bound: -70.1912455, upper bound: 70.1912455
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 450.74
Output dim: 4, lower bound: -70.1912455, upper bound: 70.1912455

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1409093, upper bound: 70.1409717
time: 103.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1409717, upper bound: 70.1409092
time: 702.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1409093, upper bound: 70.1409717
time: 102.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1409717, upper bound: 70.1409092
time: 401.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 506.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 506.08
Output dim: 4, lower bound: -70.1409093, upper bound: 70.1409717
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 506.08
Output dim: 4, lower bound: -70.1409717, upper bound: 70.1409092
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 506.08
Output dim: 4, lower bound: -70.1409093, upper bound: 70.1409717
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 506.08
Output dim: 4, lower bound: -70.1409717, upper bound: 70.1409092

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1010901, upper bound: 70.1395329
time: 100.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1394705, upper bound: 70.1011515
time: 92.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1011515, upper bound: 70.1394705
time: 110.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1395329, upper bound: 70.1010901
time: 82.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1010901, upper bound: 70.1395329
time: 114.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1394705, upper bound: 70.1011515
time: 91.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1011515, upper bound: 70.1394705
time: 111.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1395329, upper bound: 70.1010901
time: 82.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 196.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 196.39
Output dim: 4, lower bound: -70.1010901, upper bound: 70.1395329
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 196.39
Output dim: 4, lower bound: -70.1394705, upper bound: 70.1011515
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 196.39
Output dim: 4, lower bound: -70.1011515, upper bound: 70.1394705
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 196.39
Output dim: 4, lower bound: -70.1395329, upper bound: 70.1010901
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 196.39
Output dim: 4, lower bound: -70.1010901, upper bound: 70.1395329
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 196.39
Output dim: 4, lower bound: -70.1394705, upper bound: 70.1011515
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 196.39
Output dim: 4, lower bound: -70.1011515, upper bound: 70.1394705
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 196.39
Output dim: 4, lower bound: -70.1395329, upper bound: 70.1010901

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0635598, upper bound: 70.1381422
time: 99.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0996944, upper bound: 70.1022508
time: 78.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1021898, upper bound: 70.0997558
time: 112.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1380798, upper bound: 70.0636209
time: 84.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0636209, upper bound: 70.1380798
time: 118.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0997558, upper bound: 70.1021898
time: 81.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1022508, upper bound: 70.0996944
time: 78.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1381422, upper bound: 70.0635597
time: 98.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0635598, upper bound: 70.1381422
time: 99.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0996944, upper bound: 70.1022508
time: 79.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1021898, upper bound: 70.0997558
time: 110.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1380798, upper bound: 70.0636209
time: 83.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0636209, upper bound: 70.1380798
time: 119.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0997558, upper bound: 70.1021898
time: 82.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1022508, upper bound: 70.0996944
time: 77.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1381422, upper bound: 70.0635597
time: 98.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 178.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.0635598, upper bound: 70.1381422
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.0996944, upper bound: 70.1022508
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.1021898, upper bound: 70.0997558
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.1380798, upper bound: 70.0636209
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.0636209, upper bound: 70.1380798
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.0997558, upper bound: 70.1021898
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.1022508, upper bound: 70.0996944
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.1381422, upper bound: 70.0635597
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.0635598, upper bound: 70.1381422
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.0996944, upper bound: 70.1022508
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.1021898, upper bound: 70.0997558
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.1380798, upper bound: 70.0636209
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.0636209, upper bound: 70.1380798
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.0997558, upper bound: 70.1021898
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.1022508, upper bound: 70.0996944
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 178.20
Output dim: 4, lower bound: -70.1381422, upper bound: 70.0635597

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0621163, upper bound: 70.1003495
time: 151.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0256804, upper bound: 70.1367277
time: 616.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0982523, upper bound: 70.0644064
time: 134.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0618428, upper bound: 70.1008087
time: 99.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1007477, upper bound: 70.0619037
time: 95.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0643457, upper bound: 70.0983137
time: 174.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1366659, upper bound: 70.0257416
time: 173.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1002884, upper bound: 70.0621772
time: 99.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0621772, upper bound: 70.1002884
time: 92.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0257416, upper bound: 70.1366659
time: 84.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0983137, upper bound: 70.0643457
time: 87.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0619037, upper bound: 70.1007477
time: 120.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1008088, upper bound: 70.0618428
time: 106.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0644064, upper bound: 70.0982523
time: 162.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1367277, upper bound: 70.0256804
time: 86.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1003495, upper bound: 70.0621163
time: 85.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817
1: -45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743
2: -40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816
3: -50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771
4: -48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169
5: -46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748
6: -90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340
7: -54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967
8: -60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890
9: -49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356
10: -76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263
11: -80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448
12: -84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437
13: -77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543
14: -117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918
15: -60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169
16: -79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278
17: -110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991
18: -78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572
19: -57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070
20: -56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187
21: -73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277
22: -69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930
23: -61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192
24: -73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095
25: -64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981
26: -82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738
27: -69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422
28: -58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223
29: -75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847
30: -78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838
31: -80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244
32: -83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496
33: -109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238
34: -97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387
35: -91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583
36: -90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737
37: -131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123
38: -106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399
39: -118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308
40: -100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348
41: -84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357
42: -66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1476

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0621163, upper bound: 70.1003495
time: 185.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0256804, upper bound: 70.1367277
time: 618.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 807.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0621163, upper bound: 70.1003495
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0256804, upper bound: 70.1367277
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0982523, upper bound: 70.0644064
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0618428, upper bound: 70.1008087
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.1007477, upper bound: 70.0619037
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0643457, upper bound: 70.0983137
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.1366659, upper bound: 70.0257416
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.1002884, upper bound: 70.0621772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0621772, upper bound: 70.1002884
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0257416, upper bound: 70.1366659
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0983137, upper bound: 70.0643457
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0619037, upper bound: 70.1007477
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.1008088, upper bound: 70.0618428
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0644064, upper bound: 70.0982523
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.1367277, upper bound: 70.0256804
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.1003495, upper bound: 70.0621163
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0621163, upper bound: 70.1003495
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 807.04
Output dim: 4, lower bound: -70.0256804, upper bound: 70.1367277
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 807.04
Output dim: 4, lower bound: -70.0996944, upper bound: 70.1022508
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 807.04
Output dim: 4, lower bound: -70.1021898, upper bound: 70.0997558
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 807.04
Output dim: 4, lower bound: -70.1380798, upper bound: 70.0636209
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 807.04
Output dim: 4, lower bound: -70.0636209, upper bound: 70.1380798
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 807.04
Output dim: 4, lower bound: -70.0997558, upper bound: 70.1021898
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 807.04
Output dim: 4, lower bound: -70.1022508, upper bound: 70.0996944
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 807.04
Output dim: 4, lower bound: -70.1381422, upper bound: 70.0635597

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 101.85 + 7282.68 = 7384.53 seconds
