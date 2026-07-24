## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 74.4002296949
Search space: {k/256.0 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 2.75 + 101.63 = 104.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -80.9148816, upper bound: 80.9148816


# Binary Search by BASE starts (time budget: 17895.62 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=121.93731689453125
rel_dist={4: [-74.7091100354215, 74.70911000353107]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=121.93731689453125
rel_dist={4: [-70.3169252123983, 70.3169252053476]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=121.93731689453125
rel_dist={4: [-71.90757222724582, 71.90757222270969]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=121.93731689453125
rel_dist={4: [-73.36440553155583, 73.36440552353716]}

## Binary Search Result
Binary search time: 830.77 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 17064.85 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0787893, upper bound: 78.1829337
time: 93.56 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.1829334, upper bound: 78.1829337
time: 102.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 195.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 195.71
Output dim: 4, lower bound: -78.0787893, upper bound: 78.1829337
IS_A2, status: Status.UNKNOWN, split count: 1, time: 195.71
Output dim: 4, lower bound: -78.1829334, upper bound: 78.1829337

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -91.3213043, 65.7857513, -91.4896622, 65.8297272, -157.1510315, 157.2754059
1: -45.5418701, 56.0182419, -45.6620483, 56.0444260, -101.5862885, 101.6802826
2: -39.8204613, 57.0575562, -40.0057907, 57.0831985, -96.9036560, 97.0633392
3: -49.8970680, 59.1013107, -50.0993347, 59.1422462, -109.0393143, 109.2006454
4: -48.7142906, 72.9826508, -48.9216766, 73.0156403, -121.7299118, 121.9043274
5: -46.1217842, 58.0246353, -46.3402328, 58.0639458, -104.1857147, 104.3648682
6: -90.8448639, 43.7711296, -90.9129562, 43.8431740, -134.6880341, 134.6840820
7: -54.7714386, 56.6660614, -54.9336319, 56.6945648, -111.4660034, 111.5996933
8: -60.6149902, 82.6140213, -60.8093033, 82.6538086, -143.2687683, 143.4233246
9: -49.4227333, 63.5123978, -49.4784508, 63.6636848, -113.0864182, 112.9908447
10: -76.5075378, 71.8824463, -76.5874405, 72.1994781, -148.7070160, 148.4698639
11: -80.5862885, 37.4702682, -80.6512146, 37.6950340, -118.2813263, 118.1214828
12: -84.7255707, 51.1341171, -84.7720871, 51.4539680, -136.1795349, 135.9061890
13: -77.4858856, 80.6926575, -77.5684052, 80.7841492, -158.2700195, 158.2610626
14: -117.2267990, 55.5879097, -117.3127670, 55.8701401, -173.0969391, 172.9006653
15: -60.4579048, 63.1360359, -60.6139526, 63.1955643, -123.6534729, 123.7499847
16: -79.0574265, 54.7088737, -79.1500015, 54.9147339, -133.9721375, 133.8588715
17: -110.6097488, 47.7281609, -110.6607285, 47.8645630, -158.4743042, 158.3888855
18: -78.8070526, 54.2625046, -78.8685837, 54.3912735, -133.1983337, 133.1310883
19: -57.6342239, 36.0015602, -57.6920547, 36.1438560, -93.7780762, 93.6936035
20: -56.3461571, 39.7109489, -56.4062538, 39.8666687, -96.2128220, 96.1171951
21: -73.8945923, 41.4410095, -73.9520569, 41.6378708, -115.5324402, 115.3930664
22: -68.9924240, 43.9675217, -69.0435486, 44.0636444, -113.0560532, 113.0110626
23: -61.4508209, 46.6096725, -61.4972343, 46.7317924, -108.1825867, 108.1069031
24: -73.3691940, 46.1511002, -73.4364166, 46.1977005, -119.5668793, 119.5875168
25: -64.0892487, 47.4152641, -64.1333618, 47.5422401, -111.6314850, 111.5486221
26: -82.7797394, 61.6532631, -82.8443909, 61.9199905, -144.6997223, 144.4976501
27: -69.2690811, 45.9126091, -69.3682251, 45.9550247, -115.2241058, 115.2808228
28: -58.2973595, 48.7280655, -58.3477058, 48.8492203, -107.1465759, 107.0757675
29: -75.0226898, 42.1504745, -75.0632477, 42.2576370, -117.2803268, 117.2137222
30: -78.9212570, 47.7777481, -78.9728622, 47.9274483, -126.8486938, 126.7505875
31: -80.0189209, 47.7652168, -80.0942993, 47.9363213, -127.9552460, 127.8595123
32: -83.4178314, 42.6529427, -83.4712524, 42.7773972, -126.1952209, 126.1241837
33: -109.7412338, 52.0245743, -109.8924103, 52.0851135, -161.8263550, 161.9169922
34: -97.7315979, 28.4245987, -97.8313904, 28.4801559, -126.2117538, 126.2559814
35: -91.4567719, 39.5986328, -91.5555267, 39.6454468, -131.1022186, 131.1541443
36: -89.9831314, 45.4956589, -90.0425873, 45.5510864, -135.5342102, 135.5382385
37: -131.3728943, 40.3259125, -131.4534607, 40.4228439, -171.7957458, 171.7793732
38: -106.6347885, 49.6071091, -106.7509155, 49.6612320, -156.2960052, 156.3580170
39: -118.4963913, 57.1340942, -118.5973587, 57.2216721, -175.7180634, 175.7314453
40: -100.0724792, 35.2577820, -100.1530228, 35.2989044, -135.3713837, 135.4107971
41: -84.1393890, 51.1012497, -84.2078705, 51.1618767, -135.3012390, 135.3091125
42: -66.1784515, 38.0395699, -66.2317963, 38.1462288, -104.3246765, 104.2713623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1593
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
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
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
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 593
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
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1571
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
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 995
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
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1023
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
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
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
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 966
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
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0787893, upper bound: 78.0787893
time: 99.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0787893, upper bound: 78.1829337
time: 80.06 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -91.5572815, 65.9842834, -91.4820709, 65.8277817, -157.3850403, 157.4663544
1: -45.7041969, 56.2767944, -45.6576309, 56.0431938, -101.7473755, 101.9344254
2: -40.0440826, 57.3512192, -40.0001030, 57.0815964, -97.1256790, 97.3513184
3: -50.1253052, 59.4354057, -50.0918617, 59.1399879, -109.2652893, 109.5272675
4: -48.9753952, 73.3160248, -48.9153671, 73.0133667, -121.9887543, 122.2313919
5: -46.3873367, 58.3225212, -46.3347473, 58.0618019, -104.4491348, 104.6572495
6: -91.0342331, 43.8610458, -90.9093781, 43.8305435, -134.8647766, 134.7704163
7: -55.0119286, 56.8803864, -54.9271545, 56.6928635, -111.7047882, 111.8075409
8: -60.8534088, 82.9740524, -60.8029900, 82.6518402, -143.5052490, 143.7770386
9: -49.6685791, 63.7269135, -49.4753876, 63.6589775, -113.3275604, 113.2023010
10: -77.0238342, 72.2536621, -76.5835648, 72.1895142, -149.2133484, 148.8372192
11: -81.0911407, 37.7204285, -80.6484146, 37.6873283, -118.7784729, 118.3688431
12: -85.1889343, 51.4995117, -84.7696838, 51.4443665, -136.6333008, 136.2691956
13: -77.5886230, 80.8915405, -77.5580368, 80.7804108, -158.3690338, 158.4495697
14: -117.6962051, 55.9012146, -117.3087921, 55.8615570, -173.5577393, 173.2099915
15: -60.6601868, 63.3568382, -60.5971260, 63.1930771, -123.8532639, 123.9539642
16: -79.4302521, 54.9584122, -79.1450653, 54.9040451, -134.3342896, 134.1034851
17: -110.9078140, 47.9288902, -110.6582489, 47.8582382, -158.7660522, 158.5871429
18: -79.0908661, 54.4297676, -78.8660049, 54.3861160, -133.4769897, 133.2957764
19: -57.9276581, 36.1629868, -57.6896553, 36.1386299, -94.0662842, 93.8526459
20: -56.6052322, 39.8797951, -56.4038849, 39.8608818, -96.4661102, 96.2836761
21: -74.3057709, 41.6711502, -73.9491577, 41.6315727, -115.9373322, 115.6203079
22: -69.1615906, 44.1064987, -69.0408020, 44.0547485, -113.2163391, 113.1472931
23: -61.6743546, 46.7657280, -61.4950523, 46.7277145, -108.4020538, 108.2607803
24: -73.4954834, 46.2025146, -73.4307480, 46.1911087, -119.6865845, 119.6332626
25: -64.2450790, 47.5862198, -64.1307373, 47.5377274, -111.7828064, 111.7169571
26: -83.1230164, 61.9699249, -82.8408432, 61.9111328, -145.0341492, 144.8107605
27: -69.4592133, 45.9666443, -69.3638611, 45.9504738, -115.4096832, 115.3305054
28: -58.4603424, 48.8746986, -58.3456001, 48.8447456, -107.3050842, 107.2202911
29: -75.2252655, 42.2832603, -75.0605240, 42.2499504, -117.4752197, 117.3437805
30: -79.1736145, 47.9719620, -78.9699097, 47.9221878, -127.0957947, 126.9418716
31: -80.3591156, 47.9559555, -80.0915070, 47.9297333, -128.2888489, 128.0474548
32: -83.6677551, 42.8102150, -83.4683990, 42.7727432, -126.4404984, 126.2786102
33: -109.9391403, 52.2600403, -109.8865051, 52.0827065, -162.0218506, 162.1465454
34: -97.8776474, 28.5842056, -97.8271332, 28.4776382, -126.3552856, 126.4113388
35: -91.5763474, 39.7871742, -91.5498886, 39.6439743, -131.2203217, 131.3370667
36: -90.1005859, 45.6061096, -90.0382080, 45.5464134, -135.6470032, 135.6443176
37: -131.5559387, 40.4748306, -131.4476013, 40.4166946, -171.9726257, 171.9224243
38: -106.8374786, 49.7533455, -106.7439194, 49.6579742, -156.4954529, 156.4972534
39: -118.6868591, 57.3069305, -118.5920334, 57.2174835, -175.9043121, 175.8989563
40: -100.2728806, 35.3652458, -100.1491165, 35.2938690, -135.5667419, 135.5143585
41: -84.2885742, 51.1965332, -84.2042770, 51.1554489, -135.4440155, 135.4008179
42: -66.3551025, 38.1627693, -66.2287979, 38.1288490, -104.4839325, 104.3915710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1777
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
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1571
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
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 995
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
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1023
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
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1020
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
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 966
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0443440, upper bound: 78.1106828
time: 203.18 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.1575842, upper bound: 78.1575843
time: 146.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 352.46 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 352.46
Output dim: 4, lower bound: -78.0787893, upper bound: 78.0787893
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 352.46
Output dim: 4, lower bound: -78.0787893, upper bound: 78.1829337
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 352.46
Output dim: 4, lower bound: -78.0443440, upper bound: 78.1106828
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 352.46
Output dim: 4, lower bound: -78.1575842, upper bound: 78.1575843

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -91.3213043, 65.7857513, -91.3213043, 65.7857513, -157.1070557, 157.1070557
1: -45.5418701, 56.0182419, -45.5418701, 56.0182419, -101.5601044, 101.5601120
2: -39.8204613, 57.0575562, -39.8204613, 57.0575562, -96.8780060, 96.8780212
3: -49.8970680, 59.1013107, -49.8970680, 59.1013107, -108.9983826, 108.9983749
4: -48.7142906, 72.9826508, -48.7142906, 72.9826508, -121.6969299, 121.6969299
5: -46.1217842, 58.0246353, -46.1217842, 58.0246353, -104.1464081, 104.1464005
6: -90.8448639, 43.7711296, -90.8448639, 43.7711296, -134.6159973, 134.6159973
7: -54.7714386, 56.6660614, -54.7714386, 56.6660614, -111.4374924, 111.4375000
8: -60.6149902, 82.6140213, -60.6149902, 82.6140213, -143.2290039, 143.2290039
9: -49.4227333, 63.5123978, -49.4227333, 63.5123978, -112.9351349, 112.9351349
10: -76.5075378, 71.8824463, -76.5075378, 71.8824463, -148.3899841, 148.3899841
11: -80.5862885, 37.4702682, -80.5862885, 37.4702682, -118.0565567, 118.0565567
12: -84.7255707, 51.1341171, -84.7255707, 51.1341171, -135.8596802, 135.8596802
13: -77.4858856, 80.6926575, -77.4858856, 80.6926575, -158.1785431, 158.1785278
14: -117.2267990, 55.5879097, -117.2267990, 55.5879097, -172.8146973, 172.8146973
15: -60.4579048, 63.1360359, -60.4579048, 63.1360359, -123.5939331, 123.5939331
16: -79.0574265, 54.7088737, -79.0574265, 54.7088737, -133.7662964, 133.7662964
17: -110.6097488, 47.7281609, -110.6097488, 47.7281609, -158.3379059, 158.3379059
18: -78.8070526, 54.2625046, -78.8070526, 54.2625046, -133.0695343, 133.0695496
19: -57.6342239, 36.0015602, -57.6342239, 36.0015602, -93.6357880, 93.6357803
20: -56.3461571, 39.7109489, -56.3461571, 39.7109489, -96.0570908, 96.0570984
21: -73.8945923, 41.4410095, -73.8945923, 41.4410095, -115.3355942, 115.3355942
22: -68.9924240, 43.9675217, -68.9924240, 43.9675217, -112.9599304, 112.9599304
23: -61.4508209, 46.6096725, -61.4508209, 46.6096725, -108.0604706, 108.0604782
24: -73.3691940, 46.1511002, -73.3691940, 46.1511002, -119.5202789, 119.5202942
25: -64.0892487, 47.4152641, -64.0892487, 47.4152641, -111.5045166, 111.5045090
26: -82.7797394, 61.6532631, -82.7797394, 61.6532631, -144.4329987, 144.4329987
27: -69.2690811, 45.9126091, -69.2690811, 45.9126091, -115.1816864, 115.1816864
28: -58.2973595, 48.7280655, -58.2973595, 48.7280655, -107.0254211, 107.0254211
29: -75.0226898, 42.1504745, -75.0226898, 42.1504745, -117.1731644, 117.1731644
30: -78.9212570, 47.7777481, -78.9212570, 47.7777481, -126.6989975, 126.6989975
31: -80.0189209, 47.7652168, -80.0189209, 47.7652168, -127.7841339, 127.7841339
32: -83.4178314, 42.6529427, -83.4178314, 42.6529427, -126.0707703, 126.0707703
33: -109.7412338, 52.0245743, -109.7412338, 52.0245743, -161.7658081, 161.7658081
34: -97.7315979, 28.4245987, -97.7315979, 28.4245987, -126.1561890, 126.1561890
35: -91.4567719, 39.5986328, -91.4567719, 39.5986328, -131.0554047, 131.0553894
36: -89.9831314, 45.4956589, -89.9831314, 45.4956589, -135.4787903, 135.4787903
37: -131.3728943, 40.3259125, -131.3728943, 40.3259125, -171.6988068, 171.6988068
38: -106.6347885, 49.6071091, -106.6347885, 49.6071091, -156.2418976, 156.2418976
39: -118.4963913, 57.1340942, -118.4963913, 57.1340942, -175.6304932, 175.6304932
40: -100.0724792, 35.2577820, -100.0724792, 35.2577820, -135.3302612, 135.3302612
41: -84.1393890, 51.1012497, -84.1393890, 51.1012497, -135.2406311, 135.2406311
42: -66.1784515, 38.0395699, -66.1784515, 38.0395699, -104.2180176, 104.2180176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1547
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
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1022
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1555
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1587
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
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
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
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
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
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1705
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
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0936920
time: 309.99 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0936920
time: 118.54 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -91.3213043, 65.7857513, -91.5572815, 65.9842834, -157.3055878, 157.3430328
1: -45.5418701, 56.0182419, -45.7041969, 56.2767944, -101.8186493, 101.7224274
2: -39.8204613, 57.0575562, -40.0440826, 57.3512192, -97.1716766, 97.1016388
3: -49.8970680, 59.1013107, -50.1253052, 59.4354057, -109.3324738, 109.2266159
4: -48.7142906, 72.9826508, -48.9753952, 73.3160248, -122.0303192, 121.9580383
5: -46.1217842, 58.0246353, -46.3873367, 58.3225212, -104.4442902, 104.4119720
6: -90.8448639, 43.7711296, -91.0342331, 43.8610458, -134.7059021, 134.8053589
7: -54.7714386, 56.6660614, -55.0119286, 56.8803864, -111.6518250, 111.6779938
8: -60.6149902, 82.6140213, -60.8534088, 82.9740524, -143.5890503, 143.4674377
9: -49.4227333, 63.5123978, -49.6685791, 63.7269135, -113.1496429, 113.1809769
10: -76.5075378, 71.8824463, -77.0238342, 72.2536621, -148.7612000, 148.9062805
11: -80.5862885, 37.4702682, -81.0911407, 37.7204285, -118.3067169, 118.5614090
12: -84.7255707, 51.1341171, -85.1889343, 51.4995117, -136.2250671, 136.3230438
13: -77.4858856, 80.6926575, -77.5886230, 80.8915405, -158.3774109, 158.2812805
14: -117.2267990, 55.5879097, -117.6962051, 55.9012146, -173.1280212, 173.2841187
15: -60.4579048, 63.1360359, -60.6601868, 63.3568382, -123.8147430, 123.7962189
16: -79.0574265, 54.7088737, -79.4302521, 54.9584122, -134.0158386, 134.1391296
17: -110.6097488, 47.7281609, -110.9078140, 47.9288902, -158.5386353, 158.6359711
18: -78.8070526, 54.2625046, -79.0908661, 54.4297676, -133.2368011, 133.3533630
19: -57.6342239, 36.0015602, -57.9276581, 36.1629868, -93.7972107, 93.9292068
20: -56.3461571, 39.7109489, -56.6052322, 39.8797951, -96.2259445, 96.3161697
21: -73.8945923, 41.4410095, -74.3057709, 41.6711502, -115.5657425, 115.7467804
22: -68.9924240, 43.9675217, -69.1615906, 44.1064987, -113.0989075, 113.1291122
23: -61.4508209, 46.6096725, -61.6743546, 46.7657280, -108.2165375, 108.2840118
24: -73.3691940, 46.1511002, -73.4954834, 46.2025146, -119.5716934, 119.6465836
25: -64.0892487, 47.4152641, -64.2450790, 47.5862198, -111.6754532, 111.6603394
26: -82.7797394, 61.6532631, -83.1230164, 61.9699249, -144.7496643, 144.7762756
27: -69.2690811, 45.9126091, -69.4592133, 45.9666443, -115.2357178, 115.3718185
28: -58.2973595, 48.7280655, -58.4603424, 48.8746986, -107.1720581, 107.1884079
29: -75.0226898, 42.1504745, -75.2252655, 42.2832603, -117.3059540, 117.3757401
30: -78.9212570, 47.7777481, -79.1736145, 47.9719620, -126.8932190, 126.9513626
31: -80.0189209, 47.7652168, -80.3591156, 47.9559555, -127.9748688, 128.1243286
32: -83.4178314, 42.6529427, -83.6677551, 42.8102150, -126.2280426, 126.3206787
33: -109.7412338, 52.0245743, -109.9391403, 52.2600403, -162.0012512, 161.9637146
34: -97.7315979, 28.4245987, -97.8776474, 28.5842056, -126.3158035, 126.3022308
35: -91.4567719, 39.5986328, -91.5763474, 39.7871742, -131.2439423, 131.1749725
36: -89.9831314, 45.4956589, -90.1005859, 45.6061096, -135.5892334, 135.5962524
37: -131.3728943, 40.3259125, -131.5559387, 40.4748306, -171.8477173, 171.8818512
38: -106.6347885, 49.6071091, -106.8374786, 49.7533455, -156.3881378, 156.4445801
39: -118.4963913, 57.1340942, -118.6868591, 57.3069305, -175.8033142, 175.8209534
40: -100.0724792, 35.2577820, -100.2728806, 35.3652458, -135.4377136, 135.5306702
41: -84.1393890, 51.1012497, -84.2885742, 51.1965332, -135.3359222, 135.3898315
42: -66.1784515, 38.0395699, -66.3551025, 38.1627693, -104.3412170, 104.3946686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1547
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
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1022
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1555
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1587
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
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
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
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
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
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1705
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
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575845
time: 88.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575845
time: 127.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -91.5572815, 65.9842834, -91.3697281, 65.7678452, -157.3251038, 157.3540039
1: -45.7041969, 56.2767944, -45.5947571, 56.0185242, -101.7227173, 101.8715439
2: -40.0440826, 57.3512192, -39.8928757, 57.0556221, -97.0997009, 97.2440948
3: -50.1253052, 59.4354057, -49.9709587, 59.0951233, -109.2204285, 109.4063568
4: -48.9753952, 73.3160248, -48.7759171, 72.9699860, -121.9453735, 122.0919418
5: -46.3873367, 58.3225212, -46.2077179, 58.0132523, -104.4005737, 104.5302353
6: -91.0342331, 43.8610458, -90.8300934, 43.7385445, -134.7727661, 134.6911316
7: -55.0119286, 56.8803864, -54.8513412, 56.6509018, -111.6628265, 111.7317276
8: -60.8534088, 82.9740524, -60.6599960, 82.6040497, -143.4574585, 143.6340485
9: -49.6685791, 63.7269135, -49.4319191, 63.4975510, -113.1661301, 113.1588287
10: -77.0238342, 72.2536621, -76.5136108, 71.8633270, -148.8871613, 148.7672729
11: -81.0911407, 37.7204285, -80.5710373, 37.4691429, -118.5602875, 118.2914658
12: -85.1889343, 51.4995117, -84.7178955, 51.1679688, -136.3569031, 136.2174072
13: -77.5886230, 80.8915405, -77.5102386, 80.6206818, -158.2092896, 158.4017792
14: -117.6962051, 55.9012146, -117.2292252, 55.5523758, -173.2485809, 173.1304321
15: -60.6601868, 63.3568382, -60.4239159, 63.1301842, -123.7903748, 123.7807541
16: -79.4302521, 54.9584122, -79.0621109, 54.6896553, -134.1199036, 134.0205231
17: -110.9078140, 47.9288902, -110.6095810, 47.7070541, -158.6148682, 158.5384674
18: -79.0908661, 54.4297676, -78.7881622, 54.2868805, -133.3777466, 133.2179260
19: -57.9276581, 36.1629868, -57.6327629, 36.0575905, -93.9852448, 93.7957458
20: -56.6052322, 39.8797951, -56.3452225, 39.7318306, -96.3370667, 96.2250214
21: -74.3057709, 41.6711502, -73.8875732, 41.4948044, -115.8005676, 115.5587234
22: -69.1615906, 44.1064987, -68.9065628, 43.9909058, -113.1524963, 113.0130539
23: -61.6743546, 46.7657280, -61.4504814, 46.6576080, -108.3319473, 108.2162094
24: -73.4954834, 46.2025146, -73.2922287, 46.1643295, -119.6598129, 119.4947433
25: -64.2450790, 47.5862198, -64.0652618, 47.4553757, -111.7004547, 111.6514816
26: -83.1230164, 61.9699249, -82.7683640, 61.7358704, -144.8588867, 144.7382812
27: -69.4592133, 45.9666443, -69.1834869, 45.9212227, -115.3804245, 115.1501160
28: -58.4603424, 48.8746986, -58.2569160, 48.8015060, -107.2618484, 107.1316147
29: -75.2252655, 42.2832603, -74.9685364, 42.1963120, -117.4215775, 117.2518005
30: -79.1736145, 47.9719620, -78.9050293, 47.8307343, -127.0043488, 126.8769913
31: -80.3591156, 47.9559555, -80.0181808, 47.7942581, -128.1533813, 127.9741364
32: -83.6677551, 42.8102150, -83.3953781, 42.6446075, -126.3123474, 126.2055893
33: -109.9391403, 52.2600403, -109.6949158, 52.0254173, -161.9645538, 161.9549561
34: -97.8776474, 28.5842056, -97.6931839, 28.4317970, -126.3094482, 126.2773895
35: -91.5763474, 39.7871742, -91.4017715, 39.5973969, -131.1737366, 131.1889496
36: -90.1005859, 45.6061096, -89.9510040, 45.5020905, -135.6026764, 135.5571136
37: -131.5559387, 40.4748306, -131.2914124, 40.3687820, -171.9247131, 171.7662354
38: -106.8374786, 49.7533455, -106.6514587, 49.5491295, -156.3865967, 156.4047852
39: -118.6868591, 57.3069305, -118.4739380, 57.1226120, -175.8094788, 175.7808685
40: -100.2728806, 35.3652458, -100.0487137, 35.2462158, -135.5191040, 135.4139557
41: -84.2885742, 51.1965332, -84.1229401, 51.1189003, -135.4074707, 135.3194580
42: -66.3551025, 38.1627693, -66.1608276, 38.0145988, -104.3697052, 104.3235931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
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
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 945
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
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 578
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
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1476
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
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1106828
time: 101.79 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1106828
time: 133.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -91.5469055, 65.9727859, -91.5940475, 65.8692932, -157.4161987, 157.5668335
1: -45.6988411, 56.2751923, -45.7356148, 56.1705284, -101.8693695, 102.0108032
2: -40.0367813, 57.3481216, -40.0558395, 57.2595139, -97.2962952, 97.4039612
3: -50.1131554, 59.4311790, -50.1097374, 59.2917442, -109.4048920, 109.5409164
4: -48.9565544, 73.3110428, -48.9566574, 73.2093277, -122.1658783, 122.2677002
5: -46.3751755, 58.3187332, -46.3675346, 58.1828575, -104.5580292, 104.6862640
6: -91.0259933, 43.8430023, -91.0895615, 43.8445816, -134.8705750, 134.9325562
7: -55.0041428, 56.8761826, -55.0468826, 56.7558212, -111.7599487, 111.9230576
8: -60.8437881, 82.9701385, -60.8663902, 82.8761826, -143.7199707, 143.8365326
9: -49.6640625, 63.7173004, -49.7190704, 63.7041168, -113.3681717, 113.4363708
10: -77.0184021, 72.2353745, -77.1877747, 72.2230225, -149.2414246, 149.4231415
11: -81.0859375, 37.7081718, -81.2386398, 37.7092819, -118.7952118, 118.9468079
12: -85.1838074, 51.4856186, -85.2358627, 51.5087776, -136.6925812, 136.7214813
13: -77.5821075, 80.8786392, -77.6309204, 80.8897552, -158.4718628, 158.5095520
14: -117.6866989, 55.8868599, -117.8615875, 55.8895302, -173.5762329, 173.7484436
15: -60.6506119, 63.3517761, -60.6959229, 63.4217186, -124.0723267, 124.0476990
16: -79.4217834, 54.9452667, -79.5728683, 54.9414482, -134.3632355, 134.5181274
17: -110.9020309, 47.9140129, -110.9253998, 47.9392548, -158.8412781, 158.8394165
18: -79.0802383, 54.4194183, -79.0640564, 54.4322815, -133.5125122, 133.4834747
19: -57.9220924, 36.1549339, -57.8232002, 36.1631660, -94.0852585, 93.9781265
20: -56.6015320, 39.8722458, -56.6223068, 39.8787689, -96.4803009, 96.4945450
21: -74.3000259, 41.6640091, -74.2907944, 41.6773911, -115.9774170, 115.9547958
22: -69.1451569, 44.1005287, -69.1238556, 44.2005005, -113.3456497, 113.2243805
23: -61.6705551, 46.7606277, -61.6166458, 46.7708511, -108.4414062, 108.3772736
24: -73.4861450, 46.1974030, -73.5029297, 46.2124176, -119.6985626, 119.7003250
25: -64.2370453, 47.5806770, -64.1997452, 47.6025238, -111.8395691, 111.7804260
26: -83.1171112, 61.9566689, -83.0473785, 61.9770851, -145.0941772, 145.0040436
27: -69.4488678, 45.9635010, -69.4618225, 46.0043488, -115.4532166, 115.4253235
28: -58.4505959, 48.8684959, -58.3948517, 48.8729019, -107.3235016, 107.2633362
29: -75.2161407, 42.2776527, -75.1723175, 42.3245010, -117.5406342, 117.4499664
30: -79.1688690, 47.9617271, -79.0736465, 47.9691925, -127.1380463, 127.0353699
31: -80.3539124, 47.9474411, -80.2970428, 47.9552536, -128.3091736, 128.2444763
32: -83.6613770, 42.8014832, -83.7055969, 42.7950554, -126.4564285, 126.5070801
33: -109.9282990, 52.2554321, -109.9523392, 52.3461914, -162.2744904, 162.2077484
34: -97.8695145, 28.5799637, -97.8879623, 28.7153339, -126.5848389, 126.4679260
35: -91.5675659, 39.7840309, -91.5967331, 39.8908386, -131.4583740, 131.3807526
36: -90.0917664, 45.6012497, -90.1080246, 45.6243553, -135.7161255, 135.7092743
37: -131.5415192, 40.4710846, -131.5499268, 40.5084763, -172.0499878, 172.0210114
38: -106.8269577, 49.7415276, -106.8780594, 49.7437706, -156.5707245, 156.6195831
39: -118.6732254, 57.3029366, -118.7000504, 57.3199768, -175.9931946, 176.0029907
40: -100.2653046, 35.3583984, -100.2988892, 35.3835182, -135.6488190, 135.6572876
41: -84.2794495, 51.1919289, -84.2903442, 51.2121506, -135.4916077, 135.4822693
42: -66.3487015, 38.1346016, -66.4229355, 38.1013336, -104.4500351, 104.5575256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
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
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1476
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
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 955

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575845
time: 97.45 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575845
time: 127.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 227.78 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 227.78
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0936920
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 227.78
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0936920
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 227.78
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575845
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 227.78
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575845
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 227.78
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1106828
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 227.78
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1106828
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 227.78
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575845
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 227.78
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575845

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.3213043, 65.7857513, -156.9952698, 157.0469666
1: -45.4789619, 55.9934158, -45.5418701, 56.0182419, -101.4971924, 101.5352859
2: -39.7130814, 57.0314484, -39.8204613, 57.0575562, -96.7706299, 96.8519135
3: -49.7763100, 59.0561409, -49.8970680, 59.1013107, -108.8776245, 108.9532089
4: -48.5746574, 72.9390411, -48.7142906, 72.9826508, -121.5573120, 121.6533356
5: -45.9998627, 57.9758530, -46.1217842, 58.0246353, -104.0244980, 104.0976410
6: -90.7651367, 43.6791229, -90.8448639, 43.7711296, -134.5362701, 134.5239868
7: -54.6963768, 56.6240845, -54.7714386, 56.6660614, -111.3624191, 111.3955231
8: -60.4719582, 82.5659637, -60.6149902, 82.6140213, -143.0859680, 143.1809540
9: -49.3790245, 63.3509789, -49.4227333, 63.5123978, -112.8914185, 112.7737122
10: -76.4372025, 71.5561295, -76.5075378, 71.8824463, -148.3196106, 148.0636597
11: -80.5086060, 37.2519913, -80.5862885, 37.4702682, -117.9788742, 117.8382721
12: -84.6735001, 50.8577003, -84.7255707, 51.1341171, -135.8076172, 135.5832520
13: -77.4382019, 80.5325470, -77.4858856, 80.6926575, -158.1308594, 158.0184174
14: -117.1471558, 55.2790070, -117.2267990, 55.5879097, -172.7350464, 172.5057983
15: -60.2847214, 63.0727425, -60.4579048, 63.1360359, -123.4207611, 123.5306473
16: -78.9737396, 54.4944534, -79.0574265, 54.7088737, -133.6826172, 133.5518646
17: -110.5609436, 47.5792503, -110.6097488, 47.7281609, -158.2891083, 158.1889954
18: -78.7295837, 54.1633835, -78.8070526, 54.2625046, -132.9920807, 132.9704285
19: -57.5772057, 35.9203415, -57.6342239, 36.0015602, -93.5787506, 93.5545654
20: -56.2872849, 39.5818024, -56.3461571, 39.7109489, -95.9982300, 95.9279633
21: -73.8327789, 41.3040771, -73.8945923, 41.4410095, -115.2737885, 115.1986618
22: -68.8582230, 43.9032135, -68.9924240, 43.9675217, -112.8257141, 112.8956375
23: -61.4061775, 46.5395050, -61.4508209, 46.6096725, -108.0158386, 107.9903183
24: -73.2307129, 46.1248322, -73.3691940, 46.1511002, -119.3817902, 119.4940186
25: -64.0237274, 47.3331070, -64.0892487, 47.4152641, -111.4389954, 111.4223480
26: -82.7070160, 61.4796524, -82.7797394, 61.6532631, -144.3602753, 144.2593842
27: -69.0891266, 45.8833313, -69.2690811, 45.9126091, -115.0017395, 115.1524124
28: -58.2086182, 48.6852188, -58.2973595, 48.7280655, -106.9366837, 106.9825745
29: -74.9307251, 42.0965996, -75.0226898, 42.1504745, -117.0811920, 117.1192932
30: -78.8560181, 47.6862679, -78.9212570, 47.7777481, -126.6337509, 126.6075211
31: -79.9453430, 47.6296501, -80.0189209, 47.7652168, -127.7105484, 127.6485748
32: -83.3446808, 42.5247154, -83.4178314, 42.6529427, -125.9976196, 125.9425430
33: -109.5497742, 51.9672089, -109.7412338, 52.0245743, -161.5743408, 161.7084351
34: -97.5978851, 28.3786812, -97.7315979, 28.4245987, -126.0224838, 126.1102600
35: -91.3086853, 39.5517616, -91.4567719, 39.5986328, -130.9073181, 131.0085297
36: -89.8961334, 45.4528236, -89.9831314, 45.4956589, -135.3917847, 135.4359589
37: -131.2169189, 40.2779846, -131.3728943, 40.3259125, -171.5428162, 171.6508789
38: -106.5425110, 49.4991035, -106.6347885, 49.6071091, -156.1496277, 156.1338959
39: -118.3792419, 57.0401192, -118.4963913, 57.1340942, -175.5133362, 175.5364990
40: -99.9716492, 35.2111969, -100.0724792, 35.2577820, -135.2294159, 135.2836761
41: -84.0579147, 51.0648232, -84.1393890, 51.1012497, -135.1591644, 135.2042084
42: -66.1099396, 37.9253464, -66.1784515, 38.0395699, -104.1495056, 104.1037979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

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
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
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
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1415
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
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
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
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1467
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0576379, upper bound: 78.0576379
time: 114.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0576379, upper bound: 78.0936920
time: 79.03 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -91.4338989, 65.8267288, -91.3110809, 65.7746353, -157.2085266, 157.1378174
1: -45.6207848, 56.1456070, -45.5367088, 56.0166168, -101.6373978, 101.6823120
2: -39.8771133, 57.2357330, -39.8122292, 57.0545197, -96.9316254, 97.0479584
3: -49.9112701, 59.2532692, -49.8838272, 59.0970306, -109.0082779, 109.1371002
4: -48.7564049, 73.1786346, -48.6957703, 72.9776764, -121.7340851, 121.8744049
5: -46.1513596, 58.1457367, -46.1072311, 58.0208931, -104.1722412, 104.2529678
6: -91.0247345, 43.7854691, -90.8366623, 43.7534752, -134.7782135, 134.6221313
7: -54.8932571, 56.7288094, -54.7629929, 56.6619987, -111.5552521, 111.4918060
8: -60.6793213, 82.8389130, -60.6057129, 82.6100922, -143.2893982, 143.4446106
9: -49.6666260, 63.5584526, -49.4182892, 63.5031052, -113.1697235, 112.9767456
10: -77.1121521, 71.9176483, -76.5022125, 71.8646240, -148.9767761, 148.4198608
11: -81.1769180, 37.4926949, -80.5811157, 37.4581871, -118.6351013, 118.0738068
12: -85.1918182, 51.1994934, -84.7204437, 51.1205521, -136.3123779, 135.9199219
13: -77.5595093, 80.7984161, -77.4794922, 80.6796722, -158.2391815, 158.2778931
14: -117.7800751, 55.6163406, -117.2178497, 55.5735474, -173.3536072, 172.8341980
15: -60.5569153, 63.3658943, -60.4485397, 63.1309357, -123.6878510, 123.8144302
16: -79.4862671, 54.7474098, -79.0490646, 54.6961823, -134.1824341, 133.7964783
17: -110.8771286, 47.8085785, -110.6042252, 47.7132034, -158.5903015, 158.4128113
18: -79.0050049, 54.3086586, -78.7968445, 54.2520828, -133.2570648, 133.1054993
19: -57.7675934, 36.0266876, -57.6288643, 35.9938698, -93.7614594, 93.6555481
20: -56.5647888, 39.7293396, -56.3424644, 39.7035484, -96.2683411, 96.0718079
21: -74.2364197, 41.4875984, -73.8889313, 41.4341316, -115.6705475, 115.3765182
22: -69.0749054, 44.1135368, -68.9759064, 43.9617233, -113.0366287, 113.0894470
23: -61.5722313, 46.6534843, -61.4470062, 46.6048279, -108.1770554, 108.1004944
24: -73.4411926, 46.1718292, -73.3599625, 46.1458130, -119.5870056, 119.5317917
25: -64.1577988, 47.4803467, -64.0819092, 47.4097443, -111.5675430, 111.5622559
26: -82.9858704, 61.7168350, -82.7739410, 61.6387100, -144.6245728, 144.4907837
27: -69.3677979, 45.9666443, -69.2589874, 45.9095688, -115.2773666, 115.2256317
28: -58.3462563, 48.7567825, -58.2876892, 48.7218437, -107.0681000, 107.0444717
29: -75.1340790, 42.2264862, -75.0147247, 42.1450348, -117.2791138, 117.2412109
30: -79.0244064, 47.8249283, -78.9165573, 47.7680817, -126.7924652, 126.7414856
31: -80.2249451, 47.7910309, -80.0137024, 47.7569389, -127.9818878, 127.8047333
32: -83.6553040, 42.6757812, -83.4113922, 42.6445312, -126.2998199, 126.0871658
33: -109.8073120, 52.2885361, -109.7304993, 52.0204239, -161.8277283, 162.0190277
34: -97.7927094, 28.6629028, -97.7235718, 28.4207363, -126.2134476, 126.3864746
35: -91.5038528, 39.8465118, -91.4481354, 39.5957489, -131.0996094, 131.2946472
36: -90.0530090, 45.5732002, -89.9743881, 45.4905090, -135.5434875, 135.5475922
37: -131.4750671, 40.4192009, -131.3584900, 40.3223877, -171.7974243, 171.7776947
38: -106.7706680, 49.6926270, -106.6246185, 49.5952873, -156.3659515, 156.3172302
39: -118.6032639, 57.2369156, -118.4818802, 57.1296692, -175.7329407, 175.7187958
40: -100.2222214, 35.3467407, -100.0647736, 35.2511826, -135.4734039, 135.4115143
41: -84.2253418, 51.1602936, -84.1302490, 51.0970917, -135.3224182, 135.2905426
42: -66.3718262, 38.0065536, -66.1720886, 38.0117226, -104.3835373, 104.1786423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

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
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1780
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
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
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 933
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
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
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
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0936920, upper bound: 78.0576379
time: 102.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0936920, upper bound: 78.0936919
time: 84.71 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.5572815, 65.9842834, -157.1938171, 157.2829437
1: -45.4789619, 55.9934158, -45.7041969, 56.2767944, -101.7557526, 101.6976089
2: -39.7130814, 57.0314484, -40.0440826, 57.3512192, -97.0643005, 97.0755310
3: -49.7763100, 59.0561409, -50.1253052, 59.4354057, -109.2117004, 109.1814423
4: -48.5746574, 72.9390411, -48.9753952, 73.3160248, -121.8906860, 121.9144363
5: -45.9998627, 57.9758530, -46.3873367, 58.3225212, -104.3223877, 104.3631897
6: -90.7651367, 43.6791229, -91.0342331, 43.8610458, -134.6261749, 134.7133484
7: -54.6963768, 56.6240845, -55.0119286, 56.8803864, -111.5767670, 111.6360092
8: -60.4719582, 82.5659637, -60.8534088, 82.9740524, -143.4460144, 143.4193726
9: -49.3790245, 63.3509789, -49.6685791, 63.7269135, -113.1059418, 113.0195618
10: -76.4372025, 71.5561295, -77.0238342, 72.2536621, -148.6908569, 148.5799561
11: -80.5086060, 37.2519913, -81.0911407, 37.7204285, -118.2290344, 118.3431168
12: -84.6735001, 50.8577003, -85.1889343, 51.4995117, -136.1730042, 136.0466309
13: -77.4382019, 80.5325470, -77.5886230, 80.8915405, -158.3297424, 158.1211548
14: -117.1471558, 55.2790070, -117.6962051, 55.9012146, -173.0483704, 172.9752197
15: -60.2847214, 63.0727425, -60.6601868, 63.3568382, -123.6415558, 123.7329254
16: -78.9737396, 54.4944534, -79.4302521, 54.9584122, -133.9321442, 133.9246979
17: -110.5609436, 47.5792503, -110.9078140, 47.9288902, -158.4898071, 158.4870605
18: -78.7295837, 54.1633835, -79.0908661, 54.4297676, -133.1593475, 133.2542419
19: -57.5772057, 35.9203415, -57.9276581, 36.1629868, -93.7401886, 93.8479843
20: -56.2872849, 39.5818024, -56.6052322, 39.8797951, -96.1670837, 96.1870346
21: -73.8327789, 41.3040771, -74.3057709, 41.6711502, -115.5039291, 115.6098480
22: -68.8582230, 43.9032135, -69.1615906, 44.1064987, -112.9647064, 113.0648041
23: -61.4061775, 46.5395050, -61.6743546, 46.7657280, -108.1719055, 108.2138519
24: -73.2307129, 46.1248322, -73.4954834, 46.2025146, -119.4332199, 119.6203156
25: -64.0237274, 47.3331070, -64.2450790, 47.5862198, -111.6099472, 111.5781860
26: -82.7070160, 61.4796524, -83.1230164, 61.9699249, -144.6769409, 144.6026611
27: -69.0891266, 45.8833313, -69.4592133, 45.9666443, -115.0557556, 115.3425446
28: -58.2086182, 48.6852188, -58.4603424, 48.8746986, -107.0833130, 107.1455612
29: -74.9307251, 42.0965996, -75.2252655, 42.2832603, -117.2139816, 117.3218613
30: -78.8560181, 47.6862679, -79.1736145, 47.9719620, -126.8279724, 126.8598785
31: -79.9453430, 47.6296501, -80.3591156, 47.9559555, -127.9012985, 127.9887695
32: -83.3446808, 42.5247154, -83.6677551, 42.8102150, -126.1548920, 126.1924591
33: -109.5497742, 51.9672089, -109.9391403, 52.2600403, -161.8098145, 161.9063416
34: -97.5978851, 28.3786812, -97.8776474, 28.5842056, -126.1820831, 126.2563095
35: -91.3086853, 39.5517616, -91.5763474, 39.7871742, -131.0958557, 131.1281128
36: -89.8961334, 45.4528236, -90.1005859, 45.6061096, -135.5022430, 135.5534058
37: -131.2169189, 40.2779846, -131.5559387, 40.4748306, -171.6917419, 171.8339233
38: -106.5425110, 49.4991035, -106.8374786, 49.7533455, -156.2958527, 156.3365784
39: -118.3792419, 57.0401192, -118.6868591, 57.3069305, -175.6861725, 175.7269745
40: -99.9716492, 35.2111969, -100.2728806, 35.3652458, -135.3368835, 135.4840698
41: -84.0579147, 51.0648232, -84.2885742, 51.1965332, -135.2544556, 135.3533936
42: -66.1099396, 37.9253464, -66.3551025, 38.1627693, -104.2727051, 104.2804489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

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
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1774
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
type: B, layer: 1, pos: 1433
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
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 580
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
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1415
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
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 561
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
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1657
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
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
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
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1106825
time: 136.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575842
time: 152.51 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -91.4338989, 65.8267288, -91.5469055, 65.9727859, -157.4066772, 157.3736267
1: -45.6207848, 56.1456070, -45.6988411, 56.2751923, -101.8959732, 101.8444519
2: -39.8771133, 57.2357330, -40.0367813, 57.3481216, -97.2252197, 97.2725143
3: -49.9112701, 59.2532692, -50.1131554, 59.4311790, -109.3424454, 109.3664246
4: -48.7564049, 73.1786346, -48.9565544, 73.3110428, -122.0674438, 122.1351776
5: -46.1513596, 58.1457367, -46.3751755, 58.3187332, -104.4700928, 104.5209122
6: -91.0247345, 43.7854691, -91.0259933, 43.8430023, -134.8677368, 134.8114624
7: -54.8932571, 56.7288094, -55.0041428, 56.8761826, -111.7694397, 111.7329559
8: -60.6793213, 82.8389130, -60.8437881, 82.9701385, -143.6494598, 143.6827087
9: -49.6666260, 63.5584526, -49.6640625, 63.7173004, -113.3839264, 113.2225189
10: -77.1121521, 71.9176483, -77.0184021, 72.2353745, -149.3475342, 148.9360504
11: -81.1769180, 37.4926949, -81.0859375, 37.7081718, -118.8850861, 118.5786285
12: -85.1918182, 51.1994934, -85.1838074, 51.4856186, -136.6774139, 136.3833008
13: -77.5595093, 80.7984161, -77.5821075, 80.8786392, -158.4381409, 158.3805237
14: -117.7800751, 55.6163406, -117.6866989, 55.8868599, -173.6669312, 173.3030396
15: -60.5569153, 63.3658943, -60.6506119, 63.3517761, -123.9086914, 124.0165024
16: -79.4862671, 54.7474098, -79.4217834, 54.9452667, -134.4315338, 134.1691895
17: -110.8771286, 47.8085785, -110.9020309, 47.9140129, -158.7911377, 158.7106018
18: -79.0050049, 54.3086586, -79.0802383, 54.4194183, -133.4244232, 133.3888855
19: -57.7675934, 36.0266876, -57.9220924, 36.1549339, -93.9225311, 93.9487762
20: -56.5647888, 39.7293396, -56.6015320, 39.8722458, -96.4370346, 96.3308716
21: -74.2364197, 41.4875984, -74.3000259, 41.6640091, -115.9004288, 115.7876129
22: -69.0749054, 44.1135368, -69.1451569, 44.1005287, -113.1754303, 113.2586975
23: -61.5722313, 46.6534843, -61.6705551, 46.7606277, -108.3328552, 108.3240356
24: -73.4411926, 46.1718292, -73.4861450, 46.1974030, -119.6385803, 119.6579742
25: -64.1577988, 47.4803467, -64.2370453, 47.5806770, -111.7384796, 111.7173920
26: -82.9858704, 61.7168350, -83.1171112, 61.9566689, -144.9425354, 144.8339386
27: -69.3677979, 45.9666443, -69.4488678, 45.9635010, -115.3312988, 115.4155045
28: -58.3462563, 48.7567825, -58.4505959, 48.8684959, -107.2147446, 107.2073746
29: -75.1340790, 42.2264862, -75.2161407, 42.2776527, -117.4117279, 117.4426270
30: -79.0244064, 47.8249283, -79.1688690, 47.9617271, -126.9861298, 126.9937820
31: -80.2249451, 47.7910309, -80.3539124, 47.9474411, -128.1723938, 128.1449432
32: -83.6553040, 42.6757812, -83.6613770, 42.8014832, -126.4567795, 126.3371429
33: -109.8073120, 52.2885361, -109.9282990, 52.2554321, -162.0626984, 162.2168274
34: -97.7927094, 28.6629028, -97.8695145, 28.5799637, -126.3726730, 126.5324097
35: -91.5038528, 39.8465118, -91.5675659, 39.7840309, -131.2878876, 131.4140778
36: -90.0530090, 45.5732002, -90.0917664, 45.6012497, -135.6542664, 135.6649628
37: -131.4750671, 40.4192009, -131.5415192, 40.4710846, -171.9461517, 171.9607239
38: -106.7706680, 49.6926270, -106.8269577, 49.7415276, -156.5121918, 156.5195618
39: -118.6032639, 57.2369156, -118.6732254, 57.3029366, -175.9061890, 175.9101410
40: -100.2222214, 35.3467407, -100.2653046, 35.3583984, -135.5805969, 135.6120453
41: -84.2253418, 51.1602936, -84.2794495, 51.1919289, -135.4172668, 135.4397430
42: -66.3718262, 38.0065536, -66.3487015, 38.1346016, -104.5064240, 104.3552551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

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
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1774
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
type: B, layer: 1, pos: 1433
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
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 580
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
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1664
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
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1415
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
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1772
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 561
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
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1657
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
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
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
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 955

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0443440, upper bound: 78.1106825
time: 83.65 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0443440, upper bound: 78.1575841
time: 106.60 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -91.4449539, 65.9252396, -91.3697281, 65.7678452, -157.2127991, 157.2949677
1: -45.6409302, 56.2520599, -45.5947571, 56.0185242, -101.6594543, 101.8468170
2: -39.9365005, 57.3253899, -39.8928757, 57.0556221, -96.9921265, 97.2182617
3: -50.0022011, 59.3907242, -49.9709587, 59.0951233, -109.0973206, 109.3616791
4: -48.8356628, 73.2730789, -48.7759171, 72.9699860, -121.8056488, 122.0489883
5: -46.2602158, 58.2741623, -46.2077179, 58.0132523, -104.2734680, 104.4818802
6: -90.9544067, 43.7695770, -90.8300934, 43.7385445, -134.6929321, 134.5996704
7: -54.9345627, 56.8392105, -54.8513412, 56.6509018, -111.5854645, 111.6905518
8: -60.7100029, 82.9263229, -60.6599960, 82.6040497, -143.3140564, 143.5863037
9: -49.6254501, 63.5656586, -49.4319191, 63.4975510, -113.1230011, 112.9975739
10: -76.9538422, 71.9273148, -76.5136108, 71.8633270, -148.8171692, 148.4409180
11: -81.0139236, 37.5021553, -80.5710373, 37.4691429, -118.4830627, 118.0731964
12: -85.1374435, 51.2231941, -84.7178955, 51.1679688, -136.3054199, 135.9410858
13: -77.5410004, 80.7319260, -77.5102386, 80.6206818, -158.1616821, 158.2421570
14: -117.6174774, 55.5926895, -117.2292252, 55.5523758, -173.1698608, 172.8219147
15: -60.4865265, 63.2941055, -60.4239159, 63.1301842, -123.6167145, 123.7180176
16: -79.3474274, 54.7443123, -79.0621109, 54.6896553, -134.0370789, 133.8064270
17: -110.8593674, 47.7776108, -110.6095810, 47.7070541, -158.5664062, 158.3871918
18: -79.0143890, 54.3303986, -78.7881622, 54.2868805, -133.3012695, 133.1185608
19: -57.8714981, 36.0820007, -57.6327629, 36.0575905, -93.9290924, 93.7147522
20: -56.5466843, 39.7505951, -56.3452225, 39.7318306, -96.2785187, 96.0958176
21: -74.2442932, 41.5343704, -73.8875732, 41.4948044, -115.7390976, 115.4219360
22: -69.0275269, 44.0414352, -68.9065628, 43.9909058, -113.0184326, 112.9479980
23: -61.6306419, 46.6955452, -61.4504814, 46.6576080, -108.2882462, 108.1460266
24: -73.3572845, 46.1751747, -73.2922287, 46.1643295, -119.5216141, 119.4674072
25: -64.1822510, 47.5038452, -64.0652618, 47.4553757, -111.6376266, 111.5690994
26: -83.0511169, 61.7912903, -82.7683640, 61.7358704, -144.7869720, 144.5596619
27: -69.2792740, 45.9372597, -69.1834869, 45.9212227, -115.2005005, 115.1207428
28: -58.3719025, 48.8310738, -58.2569160, 48.8015060, -107.1734085, 107.0879898
29: -75.1363373, 42.2285957, -74.9685364, 42.1963120, -117.3326492, 117.1971283
30: -79.1091385, 47.8803902, -78.9050293, 47.8307343, -126.9398651, 126.7854080
31: -80.2865448, 47.8203468, -80.0181808, 47.7942581, -128.0808105, 127.8385315
32: -83.5951691, 42.6823273, -83.3953781, 42.6446075, -126.2397766, 126.0776901
33: -109.7476959, 52.2033806, -109.6949158, 52.0254173, -161.7731018, 161.8983002
34: -97.7438889, 28.5384750, -97.6931839, 28.4317970, -126.1756897, 126.2316589
35: -91.4281387, 39.7405624, -91.4017715, 39.5973969, -131.0255127, 131.1423340
36: -90.0135880, 45.5616112, -89.9510040, 45.5020905, -135.5156708, 135.5126190
37: -131.3998413, 40.4261589, -131.2914124, 40.3687820, -171.7686157, 171.7175751
38: -106.7448425, 49.6454964, -106.6514587, 49.5491295, -156.2939758, 156.2969513
39: -118.5673676, 57.2114906, -118.4739380, 57.1226120, -175.6899719, 175.6854248
40: -100.1728668, 35.3180695, -100.0487137, 35.2462158, -135.4190674, 135.3667908
41: -84.2074280, 51.1603661, -84.1229401, 51.1189003, -135.3263245, 135.2832947
42: -66.2867126, 38.0489616, -66.1608276, 38.0145988, -104.3013153, 104.2097855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 924
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
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1467
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
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.1248869, upper bound: 78.0163910
time: 106.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0223680, upper bound: 78.0163910
time: 93.94 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -91.6718216, 66.0245819, -91.3697281, 65.7678452, -157.4396667, 157.3943176
1: -45.7816429, 56.4059563, -45.5947571, 56.0185242, -101.8001709, 102.0007095
2: -40.1045761, 57.5324249, -39.8928757, 57.0556221, -97.1601868, 97.4253006
3: -50.1450996, 59.5936012, -49.9709587, 59.0951233, -109.2402039, 109.5645523
4: -49.0149269, 73.5174255, -48.7759171, 72.9699860, -121.9849091, 122.2933273
5: -46.4194183, 58.4483604, -46.2077179, 58.0132523, -104.4326706, 104.6560745
6: -91.2180634, 43.8772163, -90.8300934, 43.7385445, -134.9566040, 134.7073059
7: -55.1306305, 56.9439621, -54.8513412, 56.6509018, -111.7815323, 111.7952805
8: -60.9159775, 83.2005844, -60.6599960, 82.6040497, -143.5199890, 143.8605804
9: -49.9140930, 63.7708130, -49.4319191, 63.4975510, -113.4116440, 113.2027283
10: -77.6290283, 72.2845001, -76.5136108, 71.8633270, -149.4923553, 148.7981110
11: -81.6827850, 37.7422523, -80.5710373, 37.4691429, -119.1519165, 118.3132935
12: -85.6632385, 51.5669708, -84.7178955, 51.1679688, -136.8312073, 136.2848663
13: -77.6618881, 80.9976273, -77.5102386, 80.6206818, -158.2825623, 158.5078735
14: -118.2550354, 55.9350395, -117.2292252, 55.5523758, -173.8074036, 173.1642609
15: -60.7568092, 63.6019592, -60.4239159, 63.1301842, -123.8869934, 124.0258636
16: -79.8634109, 54.9945297, -79.0621109, 54.6896553, -134.5530701, 134.0566406
17: -111.1762238, 48.0081329, -110.6095810, 47.7070541, -158.8832703, 158.6177063
18: -79.2881393, 54.4795609, -78.7881622, 54.2868805, -133.5750122, 133.2677307
19: -58.0619698, 36.1861992, -57.6327629, 36.0575905, -94.1195602, 93.8189545
20: -56.8260345, 39.8969650, -56.3452225, 39.7318306, -96.5578613, 96.2421875
21: -74.6495361, 41.7154312, -73.8875732, 41.4948044, -116.1443329, 115.6030045
22: -69.2457123, 44.2521896, -68.9065628, 43.9909058, -113.2366180, 113.1587524
23: -61.7992744, 46.8079529, -61.4504814, 46.6576080, -108.4568710, 108.2584381
24: -73.5679626, 46.2250595, -73.2922287, 46.1643295, -119.7322922, 119.5172806
25: -64.3129425, 47.6502686, -64.0652618, 47.4553757, -111.7683105, 111.7155304
26: -83.3349991, 62.0406837, -82.7683640, 61.7358704, -145.0708618, 144.8090515
27: -69.5550232, 46.0206909, -69.1834869, 45.9212227, -115.4762421, 115.2041702
28: -58.5050430, 48.9024734, -58.2569160, 48.8015060, -107.3065491, 107.1593933
29: -75.3362579, 42.3569527, -74.9685364, 42.1963120, -117.5325699, 117.3254852
30: -79.2793808, 48.0185242, -78.9050293, 47.8307343, -127.1101074, 126.9235535
31: -80.5700684, 47.9806747, -80.0181808, 47.7942581, -128.3643188, 127.9988556
32: -83.9075317, 42.8308525, -83.3953781, 42.6446075, -126.5521393, 126.2262192
33: -110.0066528, 52.5219421, -109.6949158, 52.0254173, -162.0320740, 162.2168579
34: -97.9383545, 28.8202896, -97.6931839, 28.4317970, -126.3701477, 126.5134735
35: -91.6235504, 40.0364037, -91.4017715, 39.5973969, -131.2209473, 131.4381714
36: -90.1709671, 45.6842499, -89.9510040, 45.5020905, -135.6730499, 135.6352539
37: -131.6589966, 40.5643921, -131.2914124, 40.3687820, -172.0277710, 171.8557739
38: -106.9674683, 49.8391037, -106.6514587, 49.5491295, -156.5165710, 156.4905548
39: -118.7967148, 57.4076843, -118.4739380, 57.1226120, -175.9193268, 175.8816223
40: -100.4262314, 35.4531136, -100.0487137, 35.2462158, -135.6724548, 135.5018311
41: -84.3760071, 51.2514572, -84.1229401, 51.1189003, -135.4949036, 135.3743896
42: -66.5541000, 38.1414108, -66.1608276, 38.0145988, -104.5686951, 104.3022232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 924
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
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1467
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
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.1248869, upper bound: 78.0163910
time: 85.24 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0223680, upper bound: 78.0163910
time: 93.23 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -91.4449539, 65.9252396, -91.5940475, 65.8692932, -157.3142395, 157.5192871
1: -45.6409302, 56.2520599, -45.7356148, 56.1705284, -101.8114624, 101.9876709
2: -39.9365005, 57.3253899, -40.0558395, 57.2595139, -97.1960144, 97.3812256
3: -50.0022011, 59.3907242, -50.1097374, 59.2917442, -109.2939453, 109.5004578
4: -48.8356628, 73.2730789, -48.9566574, 73.2093277, -122.0449905, 122.2297287
5: -46.2602158, 58.2741623, -46.3675346, 58.1828575, -104.4430695, 104.6416931
6: -90.9544067, 43.7695770, -91.0895615, 43.8445816, -134.7989807, 134.8591309
7: -54.9345627, 56.8392105, -55.0468826, 56.7558212, -111.6903839, 111.8860855
8: -60.7100029, 82.9263229, -60.8663902, 82.8761826, -143.5861816, 143.7927094
9: -49.6254501, 63.5656586, -49.7190704, 63.7041168, -113.3295670, 113.2847290
10: -76.9538422, 71.9273148, -77.1877747, 72.2230225, -149.1768646, 149.1150665
11: -81.0139236, 37.5021553, -81.2386398, 37.7092819, -118.7231903, 118.7407990
12: -85.1374435, 51.2231941, -85.2358627, 51.5087776, -136.6462097, 136.4590607
13: -77.5410004, 80.7319260, -77.6309204, 80.8897552, -158.4307556, 158.3628540
14: -117.6174774, 55.5926895, -117.8615875, 55.8895302, -173.5069885, 173.4542847
15: -60.4865265, 63.2941055, -60.6959229, 63.4217186, -123.9082489, 123.9900284
16: -79.3474274, 54.7443123, -79.5728683, 54.9414482, -134.2888794, 134.3171692
17: -110.8593674, 47.7776108, -110.9253998, 47.9392548, -158.7985992, 158.7030029
18: -79.0143890, 54.3303986, -79.0640564, 54.4322815, -133.4466705, 133.3944550
19: -57.8714981, 36.0820007, -57.8232002, 36.1631660, -94.0346680, 93.9051971
20: -56.5466843, 39.7505951, -56.6223068, 39.8787689, -96.4254456, 96.3729019
21: -74.2442932, 41.5343704, -74.2907944, 41.6773911, -115.9216843, 115.8251495
22: -69.0275269, 44.0414352, -69.1238556, 44.2005005, -113.2280273, 113.1652908
23: -61.6306419, 46.6955452, -61.6166458, 46.7708511, -108.4014893, 108.3121872
24: -73.3572845, 46.1751747, -73.5029297, 46.2124176, -119.5697021, 119.6781006
25: -64.1822510, 47.5038452, -64.1997452, 47.6025238, -111.7847748, 111.7035904
26: -83.0511169, 61.7912903, -83.0473785, 61.9770851, -145.0281982, 144.8386688
27: -69.2792740, 45.9372597, -69.4618225, 46.0043488, -115.2836227, 115.3990784
28: -58.3719025, 48.8310738, -58.3948517, 48.8729019, -107.2447968, 107.2259216
29: -75.1363373, 42.2285957, -75.1723175, 42.3245010, -117.4608383, 117.4008942
30: -79.1091385, 47.8803902, -79.0736465, 47.9691925, -127.0783310, 126.9540253
31: -80.2865448, 47.8203468, -80.2970428, 47.9552536, -128.2417908, 128.1173859
32: -83.5951691, 42.6823273, -83.7055969, 42.7950554, -126.3902206, 126.3879242
33: -109.7476959, 52.2033806, -109.9523392, 52.3461914, -162.0938873, 162.1557159
34: -97.7438889, 28.5384750, -97.8879623, 28.7153339, -126.4592209, 126.4264374
35: -91.4281387, 39.7405624, -91.5967331, 39.8908386, -131.3189697, 131.3372955
36: -90.0135880, 45.5616112, -90.1080246, 45.6243553, -135.6379395, 135.6696320
37: -131.3998413, 40.4261589, -131.5499268, 40.5084763, -171.9083252, 171.9760742
38: -106.7448425, 49.6454964, -106.8780594, 49.7437706, -156.4886017, 156.5235596
39: -118.5673676, 57.2114906, -118.7000504, 57.3199768, -175.8873444, 175.9115448
40: -100.1728668, 35.3180695, -100.2988892, 35.3835182, -135.5563660, 135.6169586
41: -84.2074280, 51.1603661, -84.2903442, 51.2121506, -135.4195862, 135.4507141
42: -66.2867126, 38.0489616, -66.4229355, 38.1013336, -104.3880386, 104.4718933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1544
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
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1565
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
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1664
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
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 830
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
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 577
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
type: B, layer: 1, pos: 1577
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
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 736
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
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.1106825, upper bound: 78.0443440
time: 94.76 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0443723
time: 82.17 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -91.6718216, 66.0245819, -91.5940475, 65.8692932, -157.5411072, 157.6186218
1: -45.7816429, 56.4059563, -45.7356148, 56.1705284, -101.9521713, 102.1415634
2: -40.1045761, 57.5324249, -40.0558395, 57.2595139, -97.3640900, 97.5882645
3: -50.1450996, 59.5936012, -50.1097374, 59.2917442, -109.4368286, 109.7033386
4: -49.0149269, 73.5174255, -48.9566574, 73.2093277, -122.2242508, 122.4740677
5: -46.4194183, 58.4483604, -46.3675346, 58.1828575, -104.6022797, 104.8158875
6: -91.2180634, 43.8772163, -91.0895615, 43.8445816, -135.0626373, 134.9667664
7: -55.1306305, 56.9439621, -55.0468826, 56.7558212, -111.8864517, 111.9908218
8: -60.9159775, 83.2005844, -60.8663902, 82.8761826, -143.7921448, 144.0669708
9: -49.9140930, 63.7708130, -49.7190704, 63.7041168, -113.6182098, 113.4898834
10: -77.6290283, 72.2845001, -77.1877747, 72.2230225, -149.8520508, 149.4722595
11: -81.6827850, 37.7422523, -81.2386398, 37.7092819, -119.3920670, 118.9808960
12: -85.6632385, 51.5669708, -85.2358627, 51.5087776, -137.1719971, 136.8028259
13: -77.6618881, 80.9976273, -77.6309204, 80.8897552, -158.5516357, 158.6285400
14: -118.2550354, 55.9350395, -117.8615875, 55.8895302, -174.1445618, 173.7966309
15: -60.7568092, 63.6019592, -60.6959229, 63.4217186, -124.1785278, 124.2978668
16: -79.8634109, 54.9945297, -79.5728683, 54.9414482, -134.8048553, 134.5673828
17: -111.1762238, 48.0081329, -110.9253998, 47.9392548, -159.1154785, 158.9335327
18: -79.2881393, 54.4795609, -79.0640564, 54.4322815, -133.7204285, 133.5436096
19: -58.0619698, 36.1861992, -57.8232002, 36.1631660, -94.2251358, 94.0093918
20: -56.8260345, 39.8969650, -56.6223068, 39.8787689, -96.7047882, 96.5192719
21: -74.6495361, 41.7154312, -74.2907944, 41.6773911, -116.3269196, 116.0062180
22: -69.2457123, 44.2521896, -69.1238556, 44.2005005, -113.4462128, 113.3760452
23: -61.7992744, 46.8079529, -61.6166458, 46.7708511, -108.5701141, 108.4245987
24: -73.5679626, 46.2250595, -73.5029297, 46.2124176, -119.7803802, 119.7279739
25: -64.3129425, 47.6502686, -64.1997452, 47.6025238, -111.9154663, 111.8500137
26: -83.3349991, 62.0406837, -83.0473785, 61.9770851, -145.3120728, 145.0880585
27: -69.5550232, 46.0206909, -69.4618225, 46.0043488, -115.5593719, 115.4825134
28: -58.5050430, 48.9024734, -58.3948517, 48.8729019, -107.3779449, 107.2973099
29: -75.3362579, 42.3569527, -75.1723175, 42.3245010, -117.6607590, 117.5292664
30: -79.2793808, 48.0185242, -79.0736465, 47.9691925, -127.2485733, 127.0921707
31: -80.5700684, 47.9806747, -80.2970428, 47.9552536, -128.5253296, 128.2777100
32: -83.9075317, 42.8308525, -83.7055969, 42.7950554, -126.7025833, 126.5364456
33: -110.0066528, 52.5219421, -109.9523392, 52.3461914, -162.3528442, 162.4742737
34: -97.9383545, 28.8202896, -97.8879623, 28.7153339, -126.6536865, 126.7082443
35: -91.6235504, 40.0364037, -91.5967331, 39.8908386, -131.5143890, 131.6331177
36: -90.1709671, 45.6842499, -90.1080246, 45.6243553, -135.7953186, 135.7922668
37: -131.6589966, 40.5643921, -131.5499268, 40.5084763, -172.1674805, 172.1143188
38: -106.9674683, 49.8391037, -106.8780594, 49.7437706, -156.7112427, 156.7171631
39: -118.7967148, 57.4076843, -118.7000504, 57.3199768, -176.1166992, 176.1077271
40: -100.4262314, 35.4531136, -100.2988892, 35.3835182, -135.8097534, 135.7519989
41: -84.3760071, 51.2514572, -84.2903442, 51.2121506, -135.5881653, 135.5418091
42: -66.5541000, 38.1414108, -66.4229355, 38.1013336, -104.6554337, 104.5643311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1544
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
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1565
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
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1664
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
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 830
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
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 577
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
type: B, layer: 1, pos: 1577
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
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 736
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
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.1106825, upper bound: 78.0164120
time: 135.20 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0164120
time: 79.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 217.53 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0576379, upper bound: 78.0576379
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0576379, upper bound: 78.0936920
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0936920, upper bound: 78.0576379
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0936920, upper bound: 78.0936919
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1106825
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575842
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0443440, upper bound: 78.1106825
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0443440, upper bound: 78.1575841
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.1248869, upper bound: 78.0163910
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0223680, upper bound: 78.0163910
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.1248869, upper bound: 78.0163910
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0223680, upper bound: 78.0163910
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.1106825, upper bound: 78.0443440
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0443723
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.1106825, upper bound: 78.0164120
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 217.53
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0164120

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.2095337, 65.7256622, -156.9351959, 156.9351959
1: -45.4789619, 55.9934158, -45.4789619, 55.9934158, -101.4723587, 101.4723816
2: -39.7130814, 57.0314484, -39.7130814, 57.0314484, -96.7445221, 96.7445221
3: -49.7763100, 59.0561409, -49.7763100, 59.0561409, -108.8324432, 108.8324509
4: -48.5746574, 72.9390411, -48.5746574, 72.9390411, -121.5137024, 121.5137024
5: -45.9998627, 57.9758530, -45.9998627, 57.9758530, -103.9757156, 103.9757156
6: -90.7651367, 43.6791229, -90.7651367, 43.6791229, -134.4442444, 134.4442596
7: -54.6963768, 56.6240845, -54.6963768, 56.6240845, -111.3204498, 111.3204575
8: -60.4719582, 82.5659637, -60.4719582, 82.5659637, -143.0379181, 143.0379181
9: -49.3790245, 63.3509789, -49.3790245, 63.3509789, -112.7300034, 112.7300034
10: -76.4372025, 71.5561295, -76.4372025, 71.5561295, -147.9933319, 147.9933319
11: -80.5086060, 37.2519913, -80.5086060, 37.2519913, -117.7605972, 117.7605972
12: -84.6735001, 50.8577003, -84.6735001, 50.8577003, -135.5312042, 135.5312042
13: -77.4382019, 80.5325470, -77.4382019, 80.5325470, -157.9707336, 157.9707336
14: -117.1471558, 55.2790070, -117.1471558, 55.2790070, -172.4261475, 172.4261627
15: -60.2847214, 63.0727425, -60.2847214, 63.0727425, -123.3574524, 123.3574677
16: -78.9737396, 54.4944534, -78.9737396, 54.4944534, -133.4681702, 133.4681854
17: -110.5609436, 47.5792503, -110.5609436, 47.5792503, -158.1401978, 158.1401825
18: -78.7295837, 54.1633835, -78.7295837, 54.1633835, -132.8929443, 132.8929596
19: -57.5772057, 35.9203415, -57.5772057, 35.9203415, -93.4975433, 93.4975433
20: -56.2872849, 39.5818024, -56.2872849, 39.5818024, -95.8690872, 95.8690872
21: -73.8327789, 41.3040771, -73.8327789, 41.3040771, -115.1368561, 115.1368561
22: -68.8582230, 43.9032135, -68.8582230, 43.9032135, -112.7614365, 112.7614365
23: -61.4061775, 46.5395050, -61.4061775, 46.5395050, -107.9456787, 107.9456787
24: -73.2307129, 46.1248322, -73.2307129, 46.1248322, -119.3555450, 119.3555450
25: -64.0237274, 47.3331070, -64.0237274, 47.3331070, -111.3568344, 111.3568268
26: -82.7070160, 61.4796524, -82.7070160, 61.4796524, -144.1866608, 144.1866760
27: -69.0891266, 45.8833313, -69.0891266, 45.8833313, -114.9724579, 114.9724579
28: -58.2086182, 48.6852188, -58.2086182, 48.6852188, -106.8938370, 106.8938370
29: -74.9307251, 42.0965996, -74.9307251, 42.0965996, -117.0273132, 117.0273209
30: -78.8560181, 47.6862679, -78.8560181, 47.6862679, -126.5422821, 126.5422821
31: -79.9453430, 47.6296501, -79.9453430, 47.6296501, -127.5749969, 127.5749893
32: -83.3446808, 42.5247154, -83.3446808, 42.5247154, -125.8693848, 125.8693924
33: -109.5497742, 51.9672089, -109.5497742, 51.9672089, -161.5169678, 161.5169830
34: -97.5978851, 28.3786812, -97.5978851, 28.3786812, -125.9765625, 125.9765625
35: -91.3086853, 39.5517616, -91.3086853, 39.5517616, -130.8604431, 130.8604431
36: -89.8961334, 45.4528236, -89.8961334, 45.4528236, -135.3489532, 135.3489532
37: -131.2169189, 40.2779846, -131.2169189, 40.2779846, -171.4948883, 171.4949036
38: -106.5425110, 49.4991035, -106.5425110, 49.4991035, -156.0416107, 156.0416107
39: -118.3792419, 57.0401192, -118.3792419, 57.0401192, -175.4193573, 175.4193420
40: -99.9716492, 35.2111969, -99.9716492, 35.2111969, -135.1828308, 135.1828461
41: -84.0579147, 51.0648232, -84.0579147, 51.0648232, -135.1227417, 135.1227417
42: -66.1099396, 37.9253464, -66.1099396, 37.9253464, -104.0352783, 104.0352783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

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
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 705
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
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 673
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
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1718
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
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1463
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
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 901
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
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
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
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.9094110, upper bound: 78.0238119
time: 178.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.9094110, upper bound: 77.9626678
time: 99.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.4338989, 65.8267288, -157.0362549, 157.1595612
1: -45.4789619, 55.9934158, -45.6207848, 56.1456070, -101.6245728, 101.6141968
2: -39.7130814, 57.0314484, -39.8771133, 57.2357330, -96.9488068, 96.9085617
3: -49.7763100, 59.0561409, -49.9112701, 59.2532692, -109.0295792, 108.9674072
4: -48.5746574, 72.9390411, -48.7564049, 73.1786346, -121.7532959, 121.6954498
5: -45.9998627, 57.9758530, -46.1513596, 58.1457367, -104.1455994, 104.1272049
6: -90.7651367, 43.6791229, -91.0247345, 43.7854691, -134.5505981, 134.7038574
7: -54.6963768, 56.6240845, -54.8932571, 56.7288094, -111.4251862, 111.5173340
8: -60.4719582, 82.5659637, -60.6793213, 82.8389130, -143.3108673, 143.2452850
9: -49.3790245, 63.3509789, -49.6666260, 63.5584526, -112.9374695, 113.0176086
10: -76.4372025, 71.5561295, -77.1121521, 71.9176483, -148.3548431, 148.6682739
11: -80.5086060, 37.2519913, -81.1769180, 37.4926949, -118.0012970, 118.4289093
12: -84.6735001, 50.8577003, -85.1918182, 51.1994934, -135.8729858, 136.0495148
13: -77.4382019, 80.5325470, -77.5595093, 80.7984161, -158.2366180, 158.0920410
14: -117.1471558, 55.2790070, -117.7800751, 55.6163406, -172.7634888, 173.0590820
15: -60.2847214, 63.0727425, -60.5569153, 63.3658943, -123.6506042, 123.6296539
16: -78.9737396, 54.4944534, -79.4862671, 54.7474098, -133.7211456, 133.9807129
17: -110.5609436, 47.5792503, -110.8771286, 47.8085785, -158.3695068, 158.4563599
18: -78.7295837, 54.1633835, -79.0050049, 54.3086586, -133.0382385, 133.1683807
19: -57.5772057, 35.9203415, -57.7675934, 36.0266876, -93.6038818, 93.6879349
20: -56.2872849, 39.5818024, -56.5647888, 39.7293396, -96.0166245, 96.1465912
21: -73.8327789, 41.3040771, -74.2364197, 41.4875984, -115.3203735, 115.5404892
22: -68.8582230, 43.9032135, -69.0749054, 44.1135368, -112.9717484, 112.9781189
23: -61.4061775, 46.5395050, -61.5722313, 46.6534843, -108.0596619, 108.1117401
24: -73.2307129, 46.1248322, -73.4411926, 46.1718292, -119.4025269, 119.5660248
25: -64.0237274, 47.3331070, -64.1577988, 47.4803467, -111.5040741, 111.4909058
26: -82.7070160, 61.4796524, -82.9858704, 61.7168350, -144.4238586, 144.4655151
27: -69.0891266, 45.8833313, -69.3677979, 45.9666443, -115.0557709, 115.2511292
28: -58.2086182, 48.6852188, -58.3462563, 48.7567825, -106.9653931, 107.0314713
29: -74.9307251, 42.0965996, -75.1340790, 42.2264862, -117.1572113, 117.2306824
30: -78.8560181, 47.6862679, -79.0244064, 47.8249283, -126.6809311, 126.7106705
31: -79.9453430, 47.6296501, -80.2249451, 47.7910309, -127.7363586, 127.8545990
32: -83.3446808, 42.5247154, -83.6553040, 42.6757812, -126.0204620, 126.1800079
33: -109.5497742, 51.9672089, -109.8073120, 52.2885361, -161.8383179, 161.7745056
34: -97.5978851, 28.3786812, -97.7927094, 28.6629028, -126.2607880, 126.1713791
35: -91.3086853, 39.5517616, -91.5038528, 39.8465118, -131.1551819, 131.0556183
36: -89.8961334, 45.4528236, -90.0530090, 45.5732002, -135.4693298, 135.5058289
37: -131.2169189, 40.2779846, -131.4750671, 40.4192009, -171.6361237, 171.7530518
38: -106.5425110, 49.4991035, -106.7706680, 49.6926270, -156.2351227, 156.2697754
39: -118.3792419, 57.0401192, -118.6032639, 57.2369156, -175.6161346, 175.6433716
40: -99.9716492, 35.2111969, -100.2222214, 35.3467407, -135.3183899, 135.4334106
41: -84.0579147, 51.0648232, -84.2253418, 51.1602936, -135.2182007, 135.2901611
42: -66.1099396, 37.9253464, -66.3718262, 38.0065536, -104.1164856, 104.2971649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

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
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 705
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
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 673
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
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1718
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
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1463
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
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 901
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
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
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
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.9094110, upper bound: 78.0434464
time: 109.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.9094110, upper bound: 77.9782271
time: 108.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 220.84 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 220.84
Output dim: 4, lower bound: -77.9094110, upper bound: 78.0238119
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 220.84
Output dim: 4, lower bound: -77.9094110, upper bound: 77.9626678
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 220.84
Output dim: 4, lower bound: -77.9094110, upper bound: 78.0434464
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 220.84
Output dim: 4, lower bound: -77.9094110, upper bound: 77.9782271
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0936920, upper bound: 78.0576379
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0936920, upper bound: 78.0936919
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1106825
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0163909, upper bound: 78.1575842
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0443440, upper bound: 78.1106825
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0443440, upper bound: 78.1575841
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.1248869, upper bound: 78.0163910
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0223680, upper bound: 78.0163910
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.1248869, upper bound: 78.0163910
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0223680, upper bound: 78.0163910
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.1106825, upper bound: 78.0443440
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0443723
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.1106825, upper bound: 78.0164120
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 220.84
Output dim: 4, lower bound: -78.0163909, upper bound: 78.0164120
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=121.93731689453125
rel_dist={4: [-78.19466474792833, 78.1946647699462]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8453170, upper bound: 75.9391113
time: 102.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.9391110, upper bound: 75.9391113
time: 79.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 182.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 182.62
Output dim: 4, lower bound: -75.8453170, upper bound: 75.9391113
IS_A2, status: Status.UNKNOWN, split count: 1, time: 182.62
Output dim: 4, lower bound: -75.9391110, upper bound: 75.9391113

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -91.3213043, 65.7857513, -91.4723816, 65.8252869, -157.1465912, 157.2581329
1: -45.5418701, 56.0182419, -45.6498260, 56.0417938, -101.5836639, 101.6680679
2: -39.8204613, 57.0575562, -39.9870224, 57.0806351, -96.9010925, 97.0445786
3: -49.8970680, 59.1013107, -50.0788422, 59.1381149, -109.0351791, 109.1801529
4: -48.7142906, 72.9826508, -48.9006386, 73.0123138, -121.7265930, 121.8832855
5: -46.1217842, 58.0246353, -46.3174744, 58.0599785, -104.1817474, 104.3421021
6: -90.8448639, 43.7711296, -90.9060974, 43.8358459, -134.6807098, 134.6772156
7: -54.7714386, 56.6660614, -54.9169159, 56.6916962, -111.4631119, 111.5829773
8: -60.6149902, 82.6140213, -60.7896309, 82.6497574, -143.2647400, 143.4036560
9: -49.4227333, 63.5123978, -49.4728127, 63.6482925, -113.0710144, 112.9852066
10: -76.5075378, 71.8824463, -76.5794067, 72.1673737, -148.6749115, 148.4618530
11: -80.5862885, 37.4702682, -80.6446762, 37.6722908, -118.2585754, 118.1149445
12: -84.7255707, 51.1341171, -84.7674103, 51.4213676, -136.1469269, 135.9015198
13: -77.4858856, 80.6926575, -77.5600586, 80.7749786, -158.2608643, 158.2527161
14: -117.2267990, 55.5879097, -117.3040771, 55.8416138, -173.0684204, 172.8919830
15: -60.4579048, 63.1360359, -60.5978699, 63.1895790, -123.6474762, 123.7339020
16: -79.0574265, 54.7088737, -79.1406860, 54.8938828, -133.9513092, 133.8495636
17: -110.6097488, 47.7281609, -110.6555939, 47.8507690, -158.4604950, 158.3837585
18: -78.8070526, 54.2625046, -78.8623657, 54.3782425, -133.1852875, 133.1248627
19: -57.6342239, 36.0015602, -57.6862335, 36.1294403, -93.7636566, 93.6877899
20: -56.3461571, 39.7109489, -56.4001846, 39.8508987, -96.1970367, 96.1111298
21: -73.8945923, 41.4410095, -73.9462738, 41.6179504, -115.5125351, 115.3872757
22: -68.9924240, 43.9675217, -69.0384216, 44.0538979, -113.0463181, 113.0059357
23: -61.4508209, 46.6096725, -61.4925575, 46.7194366, -108.1702576, 108.1022263
24: -73.3691940, 46.1511002, -73.4296265, 46.1928101, -119.5619888, 119.5807266
25: -64.0892487, 47.4152641, -64.1288986, 47.5294037, -111.6186371, 111.5441589
26: -82.7797394, 61.6532631, -82.8378754, 61.8926239, -144.6723633, 144.4911194
27: -69.2690811, 45.9126091, -69.3580780, 45.9507217, -115.2198029, 115.2706909
28: -58.2973595, 48.7280655, -58.3426285, 48.8367462, -107.1341095, 107.0706940
29: -75.0226898, 42.1504745, -75.0591431, 42.2465515, -117.2692413, 117.2096176
30: -78.9212570, 47.7777481, -78.9676361, 47.9122620, -126.8335190, 126.7453766
31: -80.0189209, 47.7652168, -80.0866928, 47.9189835, -127.9379044, 127.8518982
32: -83.4178314, 42.6529427, -83.4658661, 42.7648010, -126.1826248, 126.1187973
33: -109.7412338, 52.0245743, -109.8771057, 52.0790024, -161.8202209, 161.9016724
34: -97.7315979, 28.4245987, -97.8212585, 28.4745026, -126.2061005, 126.2458496
35: -91.4567719, 39.5986328, -91.5455704, 39.6407242, -131.0974884, 131.1441956
36: -89.9831314, 45.4956589, -90.0365906, 45.5453796, -135.5285034, 135.5322571
37: -131.3728943, 40.3259125, -131.4452209, 40.4128876, -171.7857819, 171.7711334
38: -106.6347885, 49.6071091, -106.7391052, 49.6556702, -156.2904510, 156.3462219
39: -118.4963913, 57.1340942, -118.5870514, 57.2127380, -175.7091064, 175.7211456
40: -100.0724792, 35.2577820, -100.1448822, 35.2946854, -135.3671570, 135.4026642
41: -84.1393890, 51.1012497, -84.2009583, 51.1556206, -135.2950134, 135.3022156
42: -66.1784515, 38.0395699, -66.2263947, 38.1354179, -104.3138733, 104.2659607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1571
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
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 966
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
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8453170, upper bound: 75.8453170
time: 101.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8453170, upper bound: 75.9391113
time: 190.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -91.5572815, 65.9842834, -91.4787979, 65.8269043, -157.3841858, 157.4630737
1: -45.7041969, 56.2767944, -45.6555481, 56.0426178, -101.7468033, 101.9323273
2: -40.0440826, 57.3512192, -39.9973793, 57.0808411, -97.1249161, 97.3485947
3: -50.1253052, 59.4354057, -50.0882950, 59.1389160, -109.2642059, 109.5236969
4: -48.9753952, 73.3160248, -48.9123993, 73.0122375, -121.9876251, 122.2284164
5: -46.3873367, 58.3225212, -46.3321075, 58.0608559, -104.4481964, 104.6546249
6: -91.0342331, 43.8610458, -90.9077148, 43.8244934, -134.8587341, 134.7687683
7: -55.0119286, 56.8803864, -54.9241104, 56.6920509, -111.7039795, 111.8044968
8: -60.8534088, 82.9740524, -60.8000069, 82.6509247, -143.5043335, 143.7740631
9: -49.6685791, 63.7269135, -49.4739494, 63.6567459, -113.3253174, 113.2008591
10: -77.0238342, 72.2536621, -76.5818634, 72.1848145, -149.2086487, 148.8355255
11: -81.0911407, 37.7204285, -80.6470718, 37.6837006, -118.7748413, 118.3675003
12: -85.1889343, 51.4995117, -84.7684631, 51.4398232, -136.6287537, 136.2679749
13: -77.5886230, 80.8915405, -77.5531464, 80.7787018, -158.3673248, 158.4446869
14: -117.6962051, 55.9012146, -117.3068848, 55.8574905, -173.5536957, 173.2080994
15: -60.6601868, 63.3568382, -60.5892906, 63.1919098, -123.8520966, 123.9461288
16: -79.4302521, 54.9584122, -79.1427078, 54.8988953, -134.3291473, 134.1011047
17: -110.9078140, 47.9288902, -110.6571350, 47.8552437, -158.7630615, 158.5860291
18: -79.0908661, 54.4297676, -78.8648453, 54.3837090, -133.4745789, 133.2946167
19: -57.9276581, 36.1629868, -57.6885223, 36.1361618, -94.0638199, 93.8515091
20: -56.6052322, 39.8797951, -56.4027863, 39.8582268, -96.4634399, 96.2825775
21: -74.3057709, 41.6711502, -73.9477997, 41.6286125, -115.9343872, 115.6189499
22: -69.1615906, 44.1064987, -69.0395126, 44.0505295, -113.2121201, 113.1459961
23: -61.6743546, 46.7657280, -61.4940300, 46.7258034, -108.4001541, 108.2597580
24: -73.4954834, 46.2025146, -73.4280777, 46.1881981, -119.6836777, 119.6305923
25: -64.2450790, 47.5862198, -64.1295090, 47.5356216, -111.7807007, 111.7157288
26: -83.1230164, 61.9699249, -82.8392029, 61.9068832, -145.0298767, 144.8091125
27: -69.4592133, 45.9666443, -69.3618011, 45.9483795, -115.4075928, 115.3284302
28: -58.4603424, 48.8746986, -58.3446388, 48.8426208, -107.3029633, 107.2193375
29: -75.2252655, 42.2832603, -75.0592728, 42.2463150, -117.4715805, 117.3425293
30: -79.1736145, 47.9719620, -78.9685669, 47.9197235, -127.0933380, 126.9405289
31: -80.3591156, 47.9559555, -80.0901871, 47.9266052, -128.2857208, 128.0461426
32: -83.6677551, 42.8102150, -83.4670944, 42.7705421, -126.4382935, 126.2773132
33: -109.9391403, 52.2600403, -109.8837128, 52.0815735, -162.0207214, 162.1437378
34: -97.8776474, 28.5842056, -97.8251801, 28.4764900, -126.3541412, 126.4093857
35: -91.5763474, 39.7871742, -91.5472107, 39.6433105, -131.2196503, 131.3343811
36: -90.1005859, 45.6061096, -90.0360718, 45.5442200, -135.6448059, 135.6421814
37: -131.5559387, 40.4748306, -131.4449158, 40.4137421, -171.9696808, 171.9197388
38: -106.8374786, 49.7533455, -106.7410965, 49.6564713, -156.4939270, 156.4944458
39: -118.6868591, 57.3069305, -118.5895920, 57.2155151, -175.9023743, 175.8965149
40: -100.2728806, 35.3652458, -100.1473618, 35.2914581, -135.5643311, 135.5126038
41: -84.2885742, 51.1965332, -84.2026825, 51.1523857, -135.4409637, 135.3992157
42: -66.3551025, 38.1627693, -66.2273788, 38.1205330, -104.4756241, 104.3901367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1571
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
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 995
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
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1020
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
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 966
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8149566, upper bound: 75.8725438
time: 105.79 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8149566, upper bound: 75.9178818
time: 93.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 201.81 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 201.81
Output dim: 4, lower bound: -75.8453170, upper bound: 75.8453170
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 201.81
Output dim: 4, lower bound: -75.8453170, upper bound: 75.9391113
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 201.81
Output dim: 4, lower bound: -75.8149566, upper bound: 75.8725438
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 201.81
Output dim: 4, lower bound: -75.8149566, upper bound: 75.9178818

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -91.3213043, 65.7857513, -91.3213043, 65.7857513, -157.1070557, 157.1070557
1: -45.5418701, 56.0182419, -45.5418701, 56.0182419, -101.5601044, 101.5601120
2: -39.8204613, 57.0575562, -39.8204613, 57.0575562, -96.8780060, 96.8780212
3: -49.8970680, 59.1013107, -49.8970680, 59.1013107, -108.9983826, 108.9983749
4: -48.7142906, 72.9826508, -48.7142906, 72.9826508, -121.6969299, 121.6969299
5: -46.1217842, 58.0246353, -46.1217842, 58.0246353, -104.1464081, 104.1464005
6: -90.8448639, 43.7711296, -90.8448639, 43.7711296, -134.6159973, 134.6159973
7: -54.7714386, 56.6660614, -54.7714386, 56.6660614, -111.4374924, 111.4375000
8: -60.6149902, 82.6140213, -60.6149902, 82.6140213, -143.2290039, 143.2290039
9: -49.4227333, 63.5123978, -49.4227333, 63.5123978, -112.9351349, 112.9351349
10: -76.5075378, 71.8824463, -76.5075378, 71.8824463, -148.3899841, 148.3899841
11: -80.5862885, 37.4702682, -80.5862885, 37.4702682, -118.0565567, 118.0565567
12: -84.7255707, 51.1341171, -84.7255707, 51.1341171, -135.8596802, 135.8596802
13: -77.4858856, 80.6926575, -77.4858856, 80.6926575, -158.1785431, 158.1785278
14: -117.2267990, 55.5879097, -117.2267990, 55.5879097, -172.8146973, 172.8146973
15: -60.4579048, 63.1360359, -60.4579048, 63.1360359, -123.5939331, 123.5939331
16: -79.0574265, 54.7088737, -79.0574265, 54.7088737, -133.7662964, 133.7662964
17: -110.6097488, 47.7281609, -110.6097488, 47.7281609, -158.3379059, 158.3379059
18: -78.8070526, 54.2625046, -78.8070526, 54.2625046, -133.0695343, 133.0695496
19: -57.6342239, 36.0015602, -57.6342239, 36.0015602, -93.6357880, 93.6357803
20: -56.3461571, 39.7109489, -56.3461571, 39.7109489, -96.0570908, 96.0570984
21: -73.8945923, 41.4410095, -73.8945923, 41.4410095, -115.3355942, 115.3355942
22: -68.9924240, 43.9675217, -68.9924240, 43.9675217, -112.9599304, 112.9599304
23: -61.4508209, 46.6096725, -61.4508209, 46.6096725, -108.0604706, 108.0604782
24: -73.3691940, 46.1511002, -73.3691940, 46.1511002, -119.5202789, 119.5202942
25: -64.0892487, 47.4152641, -64.0892487, 47.4152641, -111.5045166, 111.5045090
26: -82.7797394, 61.6532631, -82.7797394, 61.6532631, -144.4329987, 144.4329987
27: -69.2690811, 45.9126091, -69.2690811, 45.9126091, -115.1816864, 115.1816864
28: -58.2973595, 48.7280655, -58.2973595, 48.7280655, -107.0254211, 107.0254211
29: -75.0226898, 42.1504745, -75.0226898, 42.1504745, -117.1731644, 117.1731644
30: -78.9212570, 47.7777481, -78.9212570, 47.7777481, -126.6989975, 126.6989975
31: -80.0189209, 47.7652168, -80.0189209, 47.7652168, -127.7841339, 127.7841339
32: -83.4178314, 42.6529427, -83.4178314, 42.6529427, -126.0707703, 126.0707703
33: -109.7412338, 52.0245743, -109.7412338, 52.0245743, -161.7658081, 161.7658081
34: -97.7315979, 28.4245987, -97.7315979, 28.4245987, -126.1561890, 126.1561890
35: -91.4567719, 39.5986328, -91.4567719, 39.5986328, -131.0554047, 131.0553894
36: -89.9831314, 45.4956589, -89.9831314, 45.4956589, -135.4787903, 135.4787903
37: -131.3728943, 40.3259125, -131.3728943, 40.3259125, -171.6988068, 171.6988068
38: -106.6347885, 49.6071091, -106.6347885, 49.6071091, -156.2418976, 156.2418976
39: -118.4963913, 57.1340942, -118.4963913, 57.1340942, -175.6304932, 175.6304932
40: -100.0724792, 35.2577820, -100.0724792, 35.2577820, -135.3302612, 135.3302612
41: -84.1393890, 51.1012497, -84.1393890, 51.1012497, -135.2406311, 135.2406311
42: -66.1784515, 38.0395699, -66.1784515, 38.0395699, -104.2180176, 104.2180176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1547
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
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1022
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1555
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1587
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
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
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
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
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
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1705
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
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8603185
time: 118.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8603185
time: 87.99 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -91.3213043, 65.7857513, -91.5572815, 65.9842834, -157.3055878, 157.3430328
1: -45.5418701, 56.0182419, -45.7041969, 56.2767944, -101.8186493, 101.7224274
2: -39.8204613, 57.0575562, -40.0440826, 57.3512192, -97.1716766, 97.1016388
3: -49.8970680, 59.1013107, -50.1253052, 59.4354057, -109.3324738, 109.2266159
4: -48.7142906, 72.9826508, -48.9753952, 73.3160248, -122.0303192, 121.9580383
5: -46.1217842, 58.0246353, -46.3873367, 58.3225212, -104.4442902, 104.4119720
6: -90.8448639, 43.7711296, -91.0342331, 43.8610458, -134.7059021, 134.8053589
7: -54.7714386, 56.6660614, -55.0119286, 56.8803864, -111.6518250, 111.6779938
8: -60.6149902, 82.6140213, -60.8534088, 82.9740524, -143.5890503, 143.4674377
9: -49.4227333, 63.5123978, -49.6685791, 63.7269135, -113.1496429, 113.1809769
10: -76.5075378, 71.8824463, -77.0238342, 72.2536621, -148.7612000, 148.9062805
11: -80.5862885, 37.4702682, -81.0911407, 37.7204285, -118.3067169, 118.5614090
12: -84.7255707, 51.1341171, -85.1889343, 51.4995117, -136.2250671, 136.3230438
13: -77.4858856, 80.6926575, -77.5886230, 80.8915405, -158.3774109, 158.2812805
14: -117.2267990, 55.5879097, -117.6962051, 55.9012146, -173.1280212, 173.2841187
15: -60.4579048, 63.1360359, -60.6601868, 63.3568382, -123.8147430, 123.7962189
16: -79.0574265, 54.7088737, -79.4302521, 54.9584122, -134.0158386, 134.1391296
17: -110.6097488, 47.7281609, -110.9078140, 47.9288902, -158.5386353, 158.6359711
18: -78.8070526, 54.2625046, -79.0908661, 54.4297676, -133.2368011, 133.3533630
19: -57.6342239, 36.0015602, -57.9276581, 36.1629868, -93.7972107, 93.9292068
20: -56.3461571, 39.7109489, -56.6052322, 39.8797951, -96.2259445, 96.3161697
21: -73.8945923, 41.4410095, -74.3057709, 41.6711502, -115.5657425, 115.7467804
22: -68.9924240, 43.9675217, -69.1615906, 44.1064987, -113.0989075, 113.1291122
23: -61.4508209, 46.6096725, -61.6743546, 46.7657280, -108.2165375, 108.2840118
24: -73.3691940, 46.1511002, -73.4954834, 46.2025146, -119.5716934, 119.6465836
25: -64.0892487, 47.4152641, -64.2450790, 47.5862198, -111.6754532, 111.6603394
26: -82.7797394, 61.6532631, -83.1230164, 61.9699249, -144.7496643, 144.7762756
27: -69.2690811, 45.9126091, -69.4592133, 45.9666443, -115.2357178, 115.3718185
28: -58.2973595, 48.7280655, -58.4603424, 48.8746986, -107.1720581, 107.1884079
29: -75.0226898, 42.1504745, -75.2252655, 42.2832603, -117.3059540, 117.3757401
30: -78.9212570, 47.7777481, -79.1736145, 47.9719620, -126.8932190, 126.9513626
31: -80.0189209, 47.7652168, -80.3591156, 47.9559555, -127.9748688, 128.1243286
32: -83.4178314, 42.6529427, -83.6677551, 42.8102150, -126.2280426, 126.3206787
33: -109.7412338, 52.0245743, -109.9391403, 52.2600403, -162.0012512, 161.9637146
34: -97.7315979, 28.4245987, -97.8776474, 28.5842056, -126.3158035, 126.3022308
35: -91.4567719, 39.5986328, -91.5763474, 39.7871742, -131.2439423, 131.1749725
36: -89.9831314, 45.4956589, -90.1005859, 45.6061096, -135.5892334, 135.5962524
37: -131.3728943, 40.3259125, -131.5559387, 40.4748306, -171.8477173, 171.8818512
38: -106.6347885, 49.6071091, -106.8374786, 49.7533455, -156.3881378, 156.4445801
39: -118.4963913, 57.1340942, -118.6868591, 57.3069305, -175.8033142, 175.8209534
40: -100.0724792, 35.2577820, -100.2728806, 35.3652458, -135.4377136, 135.5306702
41: -84.1393890, 51.1012497, -84.2885742, 51.1965332, -135.3359222, 135.3898315
42: -66.1784515, 38.0395699, -66.3551025, 38.1627693, -104.3412170, 104.3946686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1547
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
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1022
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1555
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1587
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
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
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
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
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
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1705
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
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7913681, upper bound: 75.9178819
time: 76.44 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7913681, upper bound: 75.9178819
time: 92.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -91.5572815, 65.9842834, -91.3664551, 65.7669525, -157.3242340, 157.3507385
1: -45.7041969, 56.2767944, -45.5926514, 56.0179520, -101.7221527, 101.8694458
2: -40.0440826, 57.3512192, -39.8901443, 57.0549088, -97.0989838, 97.2413635
3: -50.1253052, 59.4354057, -49.9674416, 59.0940666, -109.2193680, 109.4028397
4: -48.9753952, 73.3160248, -48.7729187, 72.9688568, -121.9442520, 122.0889435
5: -46.3873367, 58.3225212, -46.2050934, 58.0122795, -104.3996124, 104.5276031
6: -91.0342331, 43.8610458, -90.8284302, 43.7325096, -134.7667236, 134.6894684
7: -55.0119286, 56.8803864, -54.8482933, 56.6500854, -111.6619949, 111.7286835
8: -60.8534088, 82.9740524, -60.6570168, 82.6031342, -143.4565430, 143.6310730
9: -49.6685791, 63.7269135, -49.4304581, 63.4953232, -113.1639023, 113.1573639
10: -77.0238342, 72.2536621, -76.5118256, 71.8585663, -148.8824005, 148.7654877
11: -81.0911407, 37.7204285, -80.5697174, 37.4654808, -118.5566254, 118.2901459
12: -85.1889343, 51.4995117, -84.7167511, 51.1634331, -136.3523560, 136.2162476
13: -77.5886230, 80.8915405, -77.5053711, 80.6189575, -158.2075806, 158.3969116
14: -117.6962051, 55.9012146, -117.2273560, 55.5483170, -173.2445068, 173.1285706
15: -60.6601868, 63.3568382, -60.4160309, 63.1290283, -123.7892151, 123.7728729
16: -79.4302521, 54.9584122, -79.0597687, 54.6845627, -134.1148071, 134.0181885
17: -110.9078140, 47.9288902, -110.6084518, 47.7041054, -158.6119232, 158.5373383
18: -79.0908661, 54.4297676, -78.7870026, 54.2844429, -133.3753052, 133.2167664
19: -57.9276581, 36.1629868, -57.6316376, 36.0550957, -93.9827576, 93.7946243
20: -56.6052322, 39.8797951, -56.3441086, 39.7291336, -96.3343506, 96.2239075
21: -74.3057709, 41.6711502, -73.8861847, 41.4918327, -115.7976074, 115.5573349
22: -69.1615906, 44.1064987, -68.9052887, 43.9867096, -113.1483002, 113.0117798
23: -61.6743546, 46.7657280, -61.4494400, 46.6556816, -108.3300323, 108.2151642
24: -73.4954834, 46.2025146, -73.2895813, 46.1614227, -119.6569061, 119.4920959
25: -64.2450790, 47.5862198, -64.0640259, 47.4532700, -111.6983490, 111.6502457
26: -83.1230164, 61.9699249, -82.7667236, 61.7316399, -144.8546448, 144.7366486
27: -69.4592133, 45.9666443, -69.1814499, 45.9191360, -115.3783493, 115.1480942
28: -58.4603424, 48.8746986, -58.2559433, 48.7993851, -107.2597275, 107.1306381
29: -75.2252655, 42.2832603, -74.9672852, 42.1926613, -117.4179230, 117.2505417
30: -79.1736145, 47.9719620, -78.9036560, 47.8282700, -127.0018845, 126.8756180
31: -80.3591156, 47.9559555, -80.0168762, 47.7911720, -128.1502838, 127.9728317
32: -83.6677551, 42.8102150, -83.3940659, 42.6423798, -126.3101196, 126.2042770
33: -109.9391403, 52.2600403, -109.6922226, 52.0243378, -161.9634552, 161.9522552
34: -97.8776474, 28.5842056, -97.6912079, 28.4306774, -126.3083038, 126.2754135
35: -91.5763474, 39.7871742, -91.3990631, 39.5967255, -131.1730652, 131.1862335
36: -90.1005859, 45.6061096, -89.9489365, 45.4998550, -135.6004333, 135.5550537
37: -131.5559387, 40.4748306, -131.2887573, 40.3658600, -171.9217987, 171.7635803
38: -106.8374786, 49.7533455, -106.6485367, 49.5476570, -156.3851013, 156.4018860
39: -118.6868591, 57.3069305, -118.4711609, 57.1203766, -175.8072357, 175.7780914
40: -100.2728806, 35.3652458, -100.0469513, 35.2437897, -135.5166626, 135.4122009
41: -84.2885742, 51.1965332, -84.1212997, 51.1158485, -135.4044037, 135.3178406
42: -66.3551025, 38.1627693, -66.1593933, 38.0062904, -104.3613892, 104.3221588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
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
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 945
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
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 578
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
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1476
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
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8725438
time: 90.69 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8725438
time: 88.36 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -91.5426636, 65.9677734, -91.5907211, 65.8684464, -157.4111023, 157.5584869
1: -45.6965446, 56.2745209, -45.7334099, 56.1699677, -101.8665161, 102.0079346
2: -40.0337143, 57.3468094, -40.0530891, 57.2587700, -97.2924805, 97.3998947
3: -50.1079407, 59.4293976, -50.1071320, 59.2907181, -109.3986588, 109.5365295
4: -48.9483910, 73.3088913, -48.9536247, 73.2083130, -122.1566925, 122.2625046
5: -46.3700180, 58.3171234, -46.3649673, 58.1818314, -104.5518494, 104.6820908
6: -91.0224304, 43.8351898, -91.0879822, 43.8385544, -134.8609924, 134.9231567
7: -55.0008392, 56.8743629, -55.0437737, 56.7550507, -111.7558899, 111.9181366
8: -60.8396378, 82.9685135, -60.8633690, 82.8752518, -143.7148895, 143.8318787
9: -49.6621399, 63.7131462, -49.7176590, 63.7018394, -113.3639832, 113.4307938
10: -77.0160217, 72.2275085, -77.1859360, 72.2180634, -149.2340851, 149.4134521
11: -81.0837479, 37.7028465, -81.2372818, 37.7056427, -118.7893829, 118.9401245
12: -85.1816101, 51.4797935, -85.2347260, 51.5041885, -136.6857910, 136.7145081
13: -77.5793304, 80.8730469, -77.6260376, 80.8879700, -158.4673004, 158.4990845
14: -117.6827698, 55.8806305, -117.8597412, 55.8854904, -173.5682678, 173.7403564
15: -60.6465149, 63.3495941, -60.6879272, 63.4206047, -124.0671234, 124.0375214
16: -79.4182434, 54.9396019, -79.5705490, 54.9362984, -134.3545380, 134.5101471
17: -110.8995361, 47.9076271, -110.9242249, 47.9363213, -158.8358459, 158.8318481
18: -79.0756531, 54.4155350, -79.0629120, 54.4303360, -133.5059814, 133.4784546
19: -57.9197426, 36.1514587, -57.8220406, 36.1606293, -94.0803604, 93.9734955
20: -56.6000214, 39.8689880, -56.6212158, 39.8759918, -96.4760132, 96.4902039
21: -74.2975616, 41.6609039, -74.2894287, 41.6743431, -115.9719086, 115.9503326
22: -69.1380615, 44.0979958, -69.1225739, 44.1963882, -113.3344421, 113.2205658
23: -61.6689758, 46.7584190, -61.6156235, 46.7688904, -108.4378510, 108.3740387
24: -73.4821014, 46.1952400, -73.5002747, 46.2096176, -119.6917038, 119.6955032
25: -64.2337799, 47.5782928, -64.1984863, 47.6004181, -111.8341904, 111.7767792
26: -83.1145935, 61.9518890, -83.0457611, 61.9739571, -145.0885468, 144.9976501
27: -69.4443817, 45.9621353, -69.4597626, 46.0022087, -115.4465942, 115.4218979
28: -58.4464417, 48.8658600, -58.3938255, 48.8707733, -107.3172150, 107.2596893
29: -75.2122040, 42.2752228, -75.1711121, 42.3209229, -117.5331268, 117.4463348
30: -79.1668396, 47.9575119, -79.0722198, 47.9667168, -127.1335602, 127.0297241
31: -80.3517609, 47.9437675, -80.2957001, 47.9521179, -128.3038788, 128.2394714
32: -83.6586075, 42.7976761, -83.7043076, 42.7927856, -126.4513931, 126.5019760
33: -109.9236526, 52.2534561, -109.9495392, 52.3450394, -162.2686920, 162.2030029
34: -97.8659668, 28.5781708, -97.8860168, 28.7141380, -126.5800781, 126.4641724
35: -91.5637817, 39.7826500, -91.5940399, 39.8901482, -131.4539337, 131.3766785
36: -90.0879517, 45.5991249, -90.1060562, 45.6223145, -135.7102356, 135.7051697
37: -131.5353241, 40.4694710, -131.5472565, 40.5054855, -172.0408020, 172.0167236
38: -106.8223572, 49.7364502, -106.8745880, 49.7423782, -156.5647278, 156.6110382
39: -118.6674347, 57.3013954, -118.6975327, 57.3179779, -175.9854126, 175.9989166
40: -100.2620239, 35.3555984, -100.2971268, 35.3811378, -135.6431580, 135.6527100
41: -84.2754822, 51.1900139, -84.2887115, 51.2089767, -135.4844666, 135.4787292
42: -66.3459015, 38.1235390, -66.4215546, 38.0942612, -104.4401474, 104.5450745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1687
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
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
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
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
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
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1476
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
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 955

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7099685, upper bound: 75.8719426
time: 138.62 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8292401, upper bound: 75.8292405
time: 99.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 240.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 240.52
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8603185
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 240.52
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8603185
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 240.52
Output dim: 4, lower bound: -75.7913681, upper bound: 75.9178819
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 240.52
Output dim: 4, lower bound: -75.7913681, upper bound: 75.9178819
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 240.52
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8725438
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 240.52
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8725438
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 240.52
Output dim: 4, lower bound: -75.7099685, upper bound: 75.8719426
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 240.52
Output dim: 4, lower bound: -75.8292401, upper bound: 75.8292405

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.3213043, 65.7857513, -156.9952698, 157.0469666
1: -45.4789619, 55.9934158, -45.5418701, 56.0182419, -101.4971924, 101.5352859
2: -39.7130814, 57.0314484, -39.8204613, 57.0575562, -96.7706299, 96.8519135
3: -49.7763100, 59.0561409, -49.8970680, 59.1013107, -108.8776245, 108.9532089
4: -48.5746574, 72.9390411, -48.7142906, 72.9826508, -121.5573120, 121.6533356
5: -45.9998627, 57.9758530, -46.1217842, 58.0246353, -104.0244980, 104.0976410
6: -90.7651367, 43.6791229, -90.8448639, 43.7711296, -134.5362701, 134.5239868
7: -54.6963768, 56.6240845, -54.7714386, 56.6660614, -111.3624191, 111.3955231
8: -60.4719582, 82.5659637, -60.6149902, 82.6140213, -143.0859680, 143.1809540
9: -49.3790245, 63.3509789, -49.4227333, 63.5123978, -112.8914185, 112.7737122
10: -76.4372025, 71.5561295, -76.5075378, 71.8824463, -148.3196106, 148.0636597
11: -80.5086060, 37.2519913, -80.5862885, 37.4702682, -117.9788742, 117.8382721
12: -84.6735001, 50.8577003, -84.7255707, 51.1341171, -135.8076172, 135.5832520
13: -77.4382019, 80.5325470, -77.4858856, 80.6926575, -158.1308594, 158.0184174
14: -117.1471558, 55.2790070, -117.2267990, 55.5879097, -172.7350464, 172.5057983
15: -60.2847214, 63.0727425, -60.4579048, 63.1360359, -123.4207611, 123.5306473
16: -78.9737396, 54.4944534, -79.0574265, 54.7088737, -133.6826172, 133.5518646
17: -110.5609436, 47.5792503, -110.6097488, 47.7281609, -158.2891083, 158.1889954
18: -78.7295837, 54.1633835, -78.8070526, 54.2625046, -132.9920807, 132.9704285
19: -57.5772057, 35.9203415, -57.6342239, 36.0015602, -93.5787506, 93.5545654
20: -56.2872849, 39.5818024, -56.3461571, 39.7109489, -95.9982300, 95.9279633
21: -73.8327789, 41.3040771, -73.8945923, 41.4410095, -115.2737885, 115.1986618
22: -68.8582230, 43.9032135, -68.9924240, 43.9675217, -112.8257141, 112.8956375
23: -61.4061775, 46.5395050, -61.4508209, 46.6096725, -108.0158386, 107.9903183
24: -73.2307129, 46.1248322, -73.3691940, 46.1511002, -119.3817902, 119.4940186
25: -64.0237274, 47.3331070, -64.0892487, 47.4152641, -111.4389954, 111.4223480
26: -82.7070160, 61.4796524, -82.7797394, 61.6532631, -144.3602753, 144.2593842
27: -69.0891266, 45.8833313, -69.2690811, 45.9126091, -115.0017395, 115.1524124
28: -58.2086182, 48.6852188, -58.2973595, 48.7280655, -106.9366837, 106.9825745
29: -74.9307251, 42.0965996, -75.0226898, 42.1504745, -117.0811920, 117.1192932
30: -78.8560181, 47.6862679, -78.9212570, 47.7777481, -126.6337509, 126.6075211
31: -79.9453430, 47.6296501, -80.0189209, 47.7652168, -127.7105484, 127.6485748
32: -83.3446808, 42.5247154, -83.4178314, 42.6529427, -125.9976196, 125.9425430
33: -109.5497742, 51.9672089, -109.7412338, 52.0245743, -161.5743408, 161.7084351
34: -97.5978851, 28.3786812, -97.7315979, 28.4245987, -126.0224838, 126.1102600
35: -91.3086853, 39.5517616, -91.4567719, 39.5986328, -130.9073181, 131.0085297
36: -89.8961334, 45.4528236, -89.9831314, 45.4956589, -135.3917847, 135.4359589
37: -131.2169189, 40.2779846, -131.3728943, 40.3259125, -171.5428162, 171.6508789
38: -106.5425110, 49.4991035, -106.6347885, 49.6071091, -156.1496277, 156.1338959
39: -118.3792419, 57.0401192, -118.4963913, 57.1340942, -175.5133362, 175.5364990
40: -99.9716492, 35.2111969, -100.0724792, 35.2577820, -135.2294159, 135.2836761
41: -84.0579147, 51.0648232, -84.1393890, 51.1012497, -135.1591644, 135.2042084
42: -66.1099396, 37.9253464, -66.1784515, 38.0395699, -104.1495056, 104.1037979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
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
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
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
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1415
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
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
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
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1467
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8282968, upper bound: 75.8282968
time: 120.99 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8282968, upper bound: 75.8603185
time: 82.70 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -91.4338989, 65.8267288, -91.3071289, 65.7698364, -157.2037354, 157.1338501
1: -45.6207848, 56.1456070, -45.5345001, 56.0159073, -101.6366882, 101.6801071
2: -39.8771133, 57.2357330, -39.8087234, 57.0532036, -96.9303131, 97.0444565
3: -49.9112701, 59.2532692, -49.8781509, 59.0952034, -109.0064545, 109.1314240
4: -48.7564049, 73.1786346, -48.6876907, 72.9754944, -121.7319031, 121.8663177
5: -46.1513596, 58.1457367, -46.1015472, 58.0193062, -104.1706696, 104.2472839
6: -91.0247345, 43.7854691, -90.8331375, 43.7458267, -134.7705688, 134.6185913
7: -54.8932571, 56.7288094, -54.7594414, 56.6602173, -111.5534744, 111.4882507
8: -60.6793213, 82.8389130, -60.6016617, 82.6084290, -143.2877502, 143.4405823
9: -49.6666260, 63.5584526, -49.4163818, 63.4990463, -113.1656723, 112.9748383
10: -77.1121521, 71.9176483, -76.4999084, 71.8569336, -148.9690857, 148.4175568
11: -81.1769180, 37.4926949, -80.5789032, 37.4529533, -118.6298676, 118.0715942
12: -85.1918182, 51.1994934, -84.7182159, 51.1147919, -136.3066101, 135.9177094
13: -77.5595093, 80.7984161, -77.4767303, 80.6740723, -158.2335815, 158.2751465
14: -117.7800751, 55.6163406, -117.2140808, 55.5672836, -173.3473511, 172.8304138
15: -60.5569153, 63.3658943, -60.4444923, 63.1287155, -123.6856232, 123.8103867
16: -79.4862671, 54.7474098, -79.0455627, 54.6906700, -134.1769409, 133.7929688
17: -110.8771286, 47.8085785, -110.6018524, 47.7068024, -158.5839233, 158.4104156
18: -79.0050049, 54.3086586, -78.7924728, 54.2477684, -133.2527466, 133.1011200
19: -57.7675934, 36.0266876, -57.6265945, 35.9905128, -93.7581024, 93.6532822
20: -56.5647888, 39.7293396, -56.3409119, 39.7003784, -96.2651672, 96.0702515
21: -74.2364197, 41.4875984, -73.8864899, 41.4311104, -115.6675262, 115.3740845
22: -69.0749054, 44.1135368, -68.9687500, 43.9592361, -113.0341339, 113.0822906
23: -61.5722313, 46.6534843, -61.4454308, 46.6027412, -108.1749725, 108.0989075
24: -73.4411926, 46.1718292, -73.3559723, 46.1435432, -119.5847244, 119.5277863
25: -64.1577988, 47.4803467, -64.0788879, 47.4074326, -111.5652313, 111.5592346
26: -82.9858704, 61.7168350, -82.7714233, 61.6332054, -144.6190796, 144.4882507
27: -69.3677979, 45.9666443, -69.2546387, 45.9082031, -115.2759933, 115.2212830
28: -58.3462563, 48.7567825, -58.2835808, 48.7192001, -107.0654602, 107.0403595
29: -75.1340790, 42.2264862, -75.0112610, 42.1426544, -117.2767334, 117.2377472
30: -79.0244064, 47.8249283, -78.9145355, 47.7639732, -126.7883759, 126.7394562
31: -80.2249451, 47.7910309, -80.0115051, 47.7533340, -127.9782791, 127.8025360
32: -83.6553040, 42.6757812, -83.4086456, 42.6408882, -126.2961884, 126.0844193
33: -109.8073120, 52.2885361, -109.7258606, 52.0186539, -161.8259277, 162.0144043
34: -97.7927094, 28.6629028, -97.7200928, 28.4190559, -126.2117615, 126.3829956
35: -91.5038528, 39.8465118, -91.4443665, 39.5945129, -131.0983582, 131.2908783
36: -90.0530090, 45.5732002, -89.9705658, 45.4882545, -135.5412598, 135.5437622
37: -131.4750671, 40.4192009, -131.3522949, 40.3208809, -171.7959290, 171.7714996
38: -106.7706680, 49.6926270, -106.6201401, 49.5902863, -156.3609467, 156.3127594
39: -118.6032639, 57.2369156, -118.4757690, 57.1278839, -175.7311401, 175.7126770
40: -100.2222214, 35.3467407, -100.0614471, 35.2483902, -135.4706116, 135.4081879
41: -84.2253418, 51.1602936, -84.1262970, 51.0953636, -135.3207092, 135.2865906
42: -66.3718262, 38.0065536, -66.1693420, 37.9996414, -104.3714676, 104.1758881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1687
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
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1780
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
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
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 933
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
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
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
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 1705
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7818570, upper bound: 75.7296636
time: 80.94 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7436136, upper bound: 75.7706865
time: 132.78 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.5572815, 65.9842834, -157.1938171, 157.2829437
1: -45.4789619, 55.9934158, -45.7041969, 56.2767944, -101.7557526, 101.6976089
2: -39.7130814, 57.0314484, -40.0440826, 57.3512192, -97.0643005, 97.0755310
3: -49.7763100, 59.0561409, -50.1253052, 59.4354057, -109.2117004, 109.1814423
4: -48.5746574, 72.9390411, -48.9753952, 73.3160248, -121.8906860, 121.9144363
5: -45.9998627, 57.9758530, -46.3873367, 58.3225212, -104.3223877, 104.3631897
6: -90.7651367, 43.6791229, -91.0342331, 43.8610458, -134.6261749, 134.7133484
7: -54.6963768, 56.6240845, -55.0119286, 56.8803864, -111.5767670, 111.6360092
8: -60.4719582, 82.5659637, -60.8534088, 82.9740524, -143.4460144, 143.4193726
9: -49.3790245, 63.3509789, -49.6685791, 63.7269135, -113.1059418, 113.0195618
10: -76.4372025, 71.5561295, -77.0238342, 72.2536621, -148.6908569, 148.5799561
11: -80.5086060, 37.2519913, -81.0911407, 37.7204285, -118.2290344, 118.3431168
12: -84.6735001, 50.8577003, -85.1889343, 51.4995117, -136.1730042, 136.0466309
13: -77.4382019, 80.5325470, -77.5886230, 80.8915405, -158.3297424, 158.1211548
14: -117.1471558, 55.2790070, -117.6962051, 55.9012146, -173.0483704, 172.9752197
15: -60.2847214, 63.0727425, -60.6601868, 63.3568382, -123.6415558, 123.7329254
16: -78.9737396, 54.4944534, -79.4302521, 54.9584122, -133.9321442, 133.9246979
17: -110.5609436, 47.5792503, -110.9078140, 47.9288902, -158.4898071, 158.4870605
18: -78.7295837, 54.1633835, -79.0908661, 54.4297676, -133.1593475, 133.2542419
19: -57.5772057, 35.9203415, -57.9276581, 36.1629868, -93.7401886, 93.8479843
20: -56.2872849, 39.5818024, -56.6052322, 39.8797951, -96.1670837, 96.1870346
21: -73.8327789, 41.3040771, -74.3057709, 41.6711502, -115.5039291, 115.6098480
22: -68.8582230, 43.9032135, -69.1615906, 44.1064987, -112.9647064, 113.0648041
23: -61.4061775, 46.5395050, -61.6743546, 46.7657280, -108.1719055, 108.2138519
24: -73.2307129, 46.1248322, -73.4954834, 46.2025146, -119.4332199, 119.6203156
25: -64.0237274, 47.3331070, -64.2450790, 47.5862198, -111.6099472, 111.5781860
26: -82.7070160, 61.4796524, -83.1230164, 61.9699249, -144.6769409, 144.6026611
27: -69.0891266, 45.8833313, -69.4592133, 45.9666443, -115.0557556, 115.3425446
28: -58.2086182, 48.6852188, -58.4603424, 48.8746986, -107.0833130, 107.1455612
29: -74.9307251, 42.0965996, -75.2252655, 42.2832603, -117.2139816, 117.3218613
30: -78.8560181, 47.6862679, -79.1736145, 47.9719620, -126.8279724, 126.8598785
31: -79.9453430, 47.6296501, -80.3591156, 47.9559555, -127.9012985, 127.9887695
32: -83.3446808, 42.5247154, -83.6677551, 42.8102150, -126.1548920, 126.1924591
33: -109.5497742, 51.9672089, -109.9391403, 52.2600403, -161.8098145, 161.9063416
34: -97.5978851, 28.3786812, -97.8776474, 28.5842056, -126.1820831, 126.2563095
35: -91.3086853, 39.5517616, -91.5763474, 39.7871742, -131.0958557, 131.1281128
36: -89.8961334, 45.4528236, -90.1005859, 45.6061096, -135.5022430, 135.5534058
37: -131.2169189, 40.2779846, -131.5559387, 40.4748306, -171.6917419, 171.8339233
38: -106.5425110, 49.4991035, -106.8374786, 49.7533455, -156.2958527, 156.3365784
39: -118.3792419, 57.0401192, -118.6868591, 57.3069305, -175.6861725, 175.7269745
40: -99.9716492, 35.2111969, -100.2728806, 35.3652458, -135.3368835, 135.4840698
41: -84.0579147, 51.0648232, -84.2885742, 51.1965332, -135.2544556, 135.3533936
42: -66.1099396, 37.9253464, -66.3551025, 38.1627693, -104.2727051, 104.2804489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1774
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
type: B, layer: 1, pos: 1433
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
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 580
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
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1415
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
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 561
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
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1657
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
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
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
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8725436
time: 171.48 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7913681, upper bound: 75.9178818
time: 109.14 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -91.4338989, 65.8267288, -91.5426636, 65.9677734, -157.4016724, 157.3693848
1: -45.6207848, 56.1456070, -45.6965446, 56.2745209, -101.8953018, 101.8421478
2: -39.8771133, 57.2357330, -40.0337143, 57.3468094, -97.2239227, 97.2694397
3: -49.9112701, 59.2532692, -50.1079407, 59.4293976, -109.3406677, 109.3612061
4: -48.7564049, 73.1786346, -48.9483910, 73.3088913, -122.0652924, 122.1270142
5: -46.1513596, 58.1457367, -46.3700180, 58.3171234, -104.4684753, 104.5157547
6: -91.0247345, 43.7854691, -91.0224304, 43.8351898, -134.8598938, 134.8078766
7: -54.8932571, 56.7288094, -55.0008392, 56.8743629, -111.7676010, 111.7296448
8: -60.6793213, 82.8389130, -60.8396378, 82.9685135, -143.6478271, 143.6785583
9: -49.6666260, 63.5584526, -49.6621399, 63.7131462, -113.3797531, 113.2205963
10: -77.1121521, 71.9176483, -77.0160217, 72.2275085, -149.3396606, 148.9336700
11: -81.1769180, 37.4926949, -81.0837479, 37.7028465, -118.8797607, 118.5764465
12: -85.1918182, 51.1994934, -85.1816101, 51.4797935, -136.6716003, 136.3811035
13: -77.5595093, 80.7984161, -77.5793304, 80.8730469, -158.4325562, 158.3777466
14: -117.7800751, 55.6163406, -117.6827698, 55.8806305, -173.6606903, 173.2991028
15: -60.5569153, 63.3658943, -60.6465149, 63.3495941, -123.9065094, 124.0124054
16: -79.4862671, 54.7474098, -79.4182434, 54.9396019, -134.4258575, 134.1656494
17: -110.8771286, 47.8085785, -110.8995361, 47.9076271, -158.7847595, 158.7080994
18: -79.0050049, 54.3086586, -79.0756531, 54.4155350, -133.4205322, 133.3843079
19: -57.7675934, 36.0266876, -57.9197426, 36.1514587, -93.9190521, 93.9464264
20: -56.5647888, 39.7293396, -56.6000214, 39.8689880, -96.4337616, 96.3293610
21: -74.2364197, 41.4875984, -74.2975616, 41.6609039, -115.8973236, 115.7851562
22: -69.0749054, 44.1135368, -69.1380615, 44.0979958, -113.1728973, 113.2516022
23: -61.5722313, 46.6534843, -61.6689758, 46.7584190, -108.3306503, 108.3224487
24: -73.4411926, 46.1718292, -73.4821014, 46.1952400, -119.6364288, 119.6539230
25: -64.1577988, 47.4803467, -64.2337799, 47.5782928, -111.7360916, 111.7141266
26: -82.9858704, 61.7168350, -83.1145935, 61.9518890, -144.9377594, 144.8314209
27: -69.3677979, 45.9666443, -69.4443817, 45.9621353, -115.3299103, 115.4110260
28: -58.3462563, 48.7567825, -58.4464417, 48.8658600, -107.2121124, 107.2032166
29: -75.1340790, 42.2264862, -75.2122040, 42.2752228, -117.4093018, 117.4386902
30: -79.0244064, 47.8249283, -79.1668396, 47.9575119, -126.9819031, 126.9917679
31: -80.2249451, 47.7910309, -80.3517609, 47.9437675, -128.1687164, 128.1427917
32: -83.6553040, 42.6757812, -83.6586075, 42.7976761, -126.4529572, 126.3343735
33: -109.8073120, 52.2885361, -109.9236526, 52.2534561, -162.0607300, 162.2121887
34: -97.7927094, 28.6629028, -97.8659668, 28.5781708, -126.3708496, 126.5288620
35: -91.5038528, 39.8465118, -91.5637817, 39.7826500, -131.2864990, 131.4102936
36: -90.0530090, 45.5732002, -90.0879517, 45.5991249, -135.6521301, 135.6611481
37: -131.4750671, 40.4192009, -131.5353241, 40.4694710, -171.9445190, 171.9545288
38: -106.7706680, 49.6926270, -106.8223572, 49.7364502, -156.5071106, 156.5149841
39: -118.6032639, 57.2369156, -118.6674347, 57.3013954, -175.9046478, 175.9043579
40: -100.2222214, 35.3467407, -100.2620239, 35.3555984, -135.5778198, 135.6087646
41: -84.2253418, 51.1602936, -84.2754822, 51.1900139, -135.4153595, 135.4357758
42: -66.3718262, 38.0065536, -66.3459015, 38.1235390, -104.4953613, 104.3524399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1687
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
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
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
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1774
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
type: B, layer: 1, pos: 1433
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 580
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
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1664
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
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1415
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
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1772
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 561
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
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1657
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
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
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
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 955

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7399020, upper bound: 75.7813232
time: 93.58 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7161960, upper bound: 75.8292401
time: 150.80 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -91.4449539, 65.9252396, -91.3664551, 65.7669525, -157.2119141, 157.2916870
1: -45.6409302, 56.2520599, -45.5926514, 56.0179520, -101.6588821, 101.8447113
2: -39.9365005, 57.3253899, -39.8901443, 57.0549088, -96.9914093, 97.2155304
3: -50.0022011, 59.3907242, -49.9674416, 59.0940666, -109.0962677, 109.3581619
4: -48.8356628, 73.2730789, -48.7729187, 72.9688568, -121.8045120, 122.0459976
5: -46.2602158, 58.2741623, -46.2050934, 58.0122795, -104.2724915, 104.4792404
6: -90.9544067, 43.7695770, -90.8284302, 43.7325096, -134.6869049, 134.5979919
7: -54.9345627, 56.8392105, -54.8482933, 56.6500854, -111.5846481, 111.6875000
8: -60.7100029, 82.9263229, -60.6570168, 82.6031342, -143.3131256, 143.5833435
9: -49.6254501, 63.5656586, -49.4304581, 63.4953232, -113.1207733, 112.9961090
10: -76.9538422, 71.9273148, -76.5118256, 71.8585663, -148.8124084, 148.4391479
11: -81.0139236, 37.5021553, -80.5697174, 37.4654808, -118.4794006, 118.0718689
12: -85.1374435, 51.2231941, -84.7167511, 51.1634331, -136.3008728, 135.9399414
13: -77.5410004, 80.7319260, -77.5053711, 80.6189575, -158.1599579, 158.2373047
14: -117.6174774, 55.5926895, -117.2273560, 55.5483170, -173.1657715, 172.8200378
15: -60.4865265, 63.2941055, -60.4160309, 63.1290283, -123.6155548, 123.7101364
16: -79.3474274, 54.7443123, -79.0597687, 54.6845627, -134.0319824, 133.8040771
17: -110.8593674, 47.7776108, -110.6084518, 47.7041054, -158.5634766, 158.3860626
18: -79.0143890, 54.3303986, -78.7870026, 54.2844429, -133.2988281, 133.1174011
19: -57.8714981, 36.0820007, -57.6316376, 36.0550957, -93.9265900, 93.7136383
20: -56.5466843, 39.7505951, -56.3441086, 39.7291336, -96.2758026, 96.0947037
21: -74.2442932, 41.5343704, -73.8861847, 41.4918327, -115.7361298, 115.4205475
22: -69.0275269, 44.0414352, -68.9052887, 43.9867096, -113.0142365, 112.9467239
23: -61.6306419, 46.6955452, -61.4494400, 46.6556816, -108.2863235, 108.1449814
24: -73.3572845, 46.1751747, -73.2895813, 46.1614227, -119.5187073, 119.4647522
25: -64.1822510, 47.5038452, -64.0640259, 47.4532700, -111.6354980, 111.5678711
26: -83.0511169, 61.7912903, -82.7667236, 61.7316399, -144.7827454, 144.5580139
27: -69.2792740, 45.9372597, -69.1814499, 45.9191360, -115.1984100, 115.1187134
28: -58.3719025, 48.8310738, -58.2559433, 48.7993851, -107.1712875, 107.0870132
29: -75.1363373, 42.2285957, -74.9672852, 42.1926613, -117.3289948, 117.1958694
30: -79.1091385, 47.8803902, -78.9036560, 47.8282700, -126.9374084, 126.7840195
31: -80.2865448, 47.8203468, -80.0168762, 47.7911720, -128.0777130, 127.8372192
32: -83.5951691, 42.6823273, -83.3940659, 42.6423798, -126.2375488, 126.0763779
33: -109.7476959, 52.2033806, -109.6922226, 52.0243378, -161.7720184, 161.8955994
34: -97.7438889, 28.5384750, -97.6912079, 28.4306774, -126.1745682, 126.2296753
35: -91.4281387, 39.7405624, -91.3990631, 39.5967255, -131.0248718, 131.1396179
36: -90.0135880, 45.5616112, -89.9489365, 45.4998550, -135.5134430, 135.5105438
37: -131.3998413, 40.4261589, -131.2887573, 40.3658600, -171.7657013, 171.7149200
38: -106.7448425, 49.6454964, -106.6485367, 49.5476570, -156.2924805, 156.2940369
39: -118.5673676, 57.2114906, -118.4711609, 57.1203766, -175.6877441, 175.6826477
40: -100.1728668, 35.3180695, -100.0469513, 35.2437897, -135.4166565, 135.3650208
41: -84.2074280, 51.1603661, -84.1212997, 51.1158485, -135.3232727, 135.2816620
42: -66.2867126, 38.0489616, -66.1593933, 38.0062904, -104.2929993, 104.2083588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 924
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
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1467
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
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
time: 91.19 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
time: 109.35 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -91.6718216, 66.0245819, -91.3664551, 65.7669525, -157.4387817, 157.3910370
1: -45.7816429, 56.4059563, -45.5926514, 56.0179520, -101.7995911, 101.9986038
2: -40.1045761, 57.5324249, -39.8901443, 57.0549088, -97.1594849, 97.4225693
3: -50.1450996, 59.5936012, -49.9674416, 59.0940666, -109.2391663, 109.5610352
4: -49.0149269, 73.5174255, -48.7729187, 72.9688568, -121.9837723, 122.2903442
5: -46.4194183, 58.4483604, -46.2050934, 58.0122795, -104.4317017, 104.6534424
6: -91.2180634, 43.8772163, -90.8284302, 43.7325096, -134.9505615, 134.7056427
7: -55.1306305, 56.9439621, -54.8482933, 56.6500854, -111.7807159, 111.7922363
8: -60.9159775, 83.2005844, -60.6570168, 82.6031342, -143.5191040, 143.8576050
9: -49.9140930, 63.7708130, -49.4304581, 63.4953232, -113.4094162, 113.2012711
10: -77.6290283, 72.2845001, -76.5118256, 71.8585663, -149.4875946, 148.7963257
11: -81.6827850, 37.7422523, -80.5697174, 37.4654808, -119.1482697, 118.3119659
12: -85.6632385, 51.5669708, -84.7167511, 51.1634331, -136.8266602, 136.2837219
13: -77.6618881, 80.9976273, -77.5053711, 80.6189575, -158.2808533, 158.5029907
14: -118.2550354, 55.9350395, -117.2273560, 55.5483170, -173.8033447, 173.1623993
15: -60.7568092, 63.6019592, -60.4160309, 63.1290283, -123.8858337, 124.0179749
16: -79.8634109, 54.9945297, -79.0597687, 54.6845627, -134.5479736, 134.0542908
17: -111.1762238, 48.0081329, -110.6084518, 47.7041054, -158.8803101, 158.6165771
18: -79.2881393, 54.4795609, -78.7870026, 54.2844429, -133.5725708, 133.2665710
19: -58.0619698, 36.1861992, -57.6316376, 36.0550957, -94.1170654, 93.8178329
20: -56.8260345, 39.8969650, -56.3441086, 39.7291336, -96.5551605, 96.2410736
21: -74.6495361, 41.7154312, -73.8861847, 41.4918327, -116.1413498, 115.6016159
22: -69.2457123, 44.2521896, -68.9052887, 43.9867096, -113.2324142, 113.1574783
23: -61.7992744, 46.8079529, -61.4494400, 46.6556816, -108.4549561, 108.2573929
24: -73.5679626, 46.2250595, -73.2895813, 46.1614227, -119.7293854, 119.5146408
25: -64.3129425, 47.6502686, -64.0640259, 47.4532700, -111.7662048, 111.7142944
26: -83.3349991, 62.0406837, -82.7667236, 61.7316399, -145.0666199, 144.8074036
27: -69.5550232, 46.0206909, -69.1814499, 45.9191360, -115.4741592, 115.2021408
28: -58.5050430, 48.9024734, -58.2559433, 48.7993851, -107.3044281, 107.1584091
29: -75.3362579, 42.3569527, -74.9672852, 42.1926613, -117.5289154, 117.3242340
30: -79.2793808, 48.0185242, -78.9036560, 47.8282700, -127.1076279, 126.9221649
31: -80.5700684, 47.9806747, -80.0168762, 47.7911720, -128.3612366, 127.9975510
32: -83.9075317, 42.8308525, -83.3940659, 42.6423798, -126.5499115, 126.2249069
33: -110.0066528, 52.5219421, -109.6922226, 52.0243378, -162.0309906, 162.2141724
34: -97.9383545, 28.8202896, -97.6912079, 28.4306774, -126.3690338, 126.5114975
35: -91.6235504, 40.0364037, -91.3990631, 39.5967255, -131.2202759, 131.4354553
36: -90.1709671, 45.6842499, -89.9489365, 45.4998550, -135.6708221, 135.6331787
37: -131.6589966, 40.5643921, -131.2887573, 40.3658600, -172.0248566, 171.8531494
38: -106.9674683, 49.8391037, -106.6485367, 49.5476570, -156.5151062, 156.4876404
39: -118.7967148, 57.4076843, -118.4711609, 57.1203766, -175.9170837, 175.8788452
40: -100.4262314, 35.4531136, -100.0469513, 35.2437897, -135.6700134, 135.5000610
41: -84.3760071, 51.2514572, -84.1212997, 51.1158485, -135.4918518, 135.3727570
42: -66.5541000, 38.1414108, -66.1593933, 38.0062904, -104.5603943, 104.3008041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 924
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
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1467
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
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
time: 136.98 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
time: 106.94 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -91.4620056, 65.9388809, -91.5907211, 65.8684464, -157.3304443, 157.5296021
1: -45.6329689, 56.2563019, -45.7334099, 56.1699677, -101.8029327, 101.9897156
2: -39.9465523, 57.3267784, -40.0530891, 57.2587700, -97.2053146, 97.3798599
3: -49.9987411, 59.4021721, -50.1071320, 59.2907181, -109.2894592, 109.5093079
4: -48.8431206, 73.2878723, -48.9536247, 73.2083130, -122.0514221, 122.2415009
5: -46.2713470, 58.2922287, -46.3649673, 58.1818314, -104.4531784, 104.6571960
6: -90.9853516, 43.8059158, -91.0879822, 43.8385544, -134.8238983, 134.8938904
7: -54.9268684, 56.8572235, -55.0437737, 56.7550507, -111.6819153, 111.9009933
8: -60.7566261, 82.9392242, -60.8633690, 82.8752518, -143.6318817, 143.8025818
9: -49.6263390, 63.6775551, -49.7176590, 63.7018394, -113.3281784, 113.3952179
10: -76.9733734, 72.0603790, -77.1859360, 72.2180634, -149.1914368, 149.2463074
11: -81.0439529, 37.5440903, -81.2372818, 37.7056427, -118.7495880, 118.7813721
12: -85.1501770, 51.3465118, -85.2347260, 51.5041885, -136.6543579, 136.5812378
13: -77.4692917, 80.8197937, -77.6260376, 80.8879700, -158.3572693, 158.4458313
14: -117.6221924, 55.6867218, -117.8597412, 55.8854904, -173.5076599, 173.5464630
15: -60.5453796, 63.3150101, -60.6879272, 63.4206047, -123.9659882, 124.0029373
16: -79.3568954, 54.8371239, -79.5705490, 54.9362984, -134.2931824, 134.4076691
17: -110.8576584, 47.7862778, -110.9242249, 47.9363213, -158.7939758, 158.7104950
18: -79.0279083, 54.2972260, -79.0629120, 54.4303360, -133.4582520, 133.3601379
19: -57.8840981, 36.0474968, -57.8220406, 36.1606293, -94.0447159, 93.8695374
20: -56.5632935, 39.7703552, -56.6212158, 39.8759918, -96.4392853, 96.3915710
21: -74.2595215, 41.5234451, -74.2894287, 41.6743431, -115.9338531, 115.8128738
22: -69.1062317, 44.0364304, -69.1225739, 44.1963882, -113.3026199, 113.1590042
23: -61.6394577, 46.6505547, -61.6156235, 46.7688904, -108.4083481, 108.2661743
24: -73.4467163, 46.1082840, -73.5002747, 46.2096176, -119.6563187, 119.6085510
25: -64.2019958, 47.4798470, -64.1984863, 47.6004181, -111.8024063, 111.6783295
26: -83.0730667, 61.7901001, -83.0457611, 61.9739571, -145.0470276, 144.8358612
27: -69.4006805, 45.8975105, -69.4597626, 46.0022087, -115.4028931, 115.3572693
28: -58.4127998, 48.7796021, -58.3938255, 48.8707733, -107.2835541, 107.1734314
29: -75.1839142, 42.2011223, -75.1711121, 42.3209229, -117.5048141, 117.3722305
30: -79.1314850, 47.8296890, -79.0722198, 47.9667168, -127.0981979, 126.9019089
31: -80.3045807, 47.8017273, -80.2957001, 47.9521179, -128.2566986, 128.0974274
32: -83.6178665, 42.7633286, -83.7043076, 42.7927856, -126.4106522, 126.4676361
33: -109.8238144, 52.2214661, -109.9495392, 52.3450394, -162.1688385, 162.1710052
34: -97.7998886, 28.5434456, -97.8860168, 28.7141380, -126.5140076, 126.4294510
35: -91.4968262, 39.7579079, -91.5940399, 39.8901482, -131.3869781, 131.3519440
36: -90.0222931, 45.5719147, -90.1060562, 45.6223145, -135.6445923, 135.6779785
37: -131.4814758, 40.4150848, -131.5472565, 40.5054855, -171.9869385, 171.9623413
38: -106.7445526, 49.7022476, -106.8745880, 49.7423782, -156.4869385, 156.5768280
39: -118.5837631, 57.2702065, -118.6975327, 57.3179779, -175.9017334, 175.9677429
40: -100.2131348, 35.3368568, -100.2971268, 35.3811378, -135.5942688, 135.6339722
41: -84.2212830, 51.1594467, -84.2887115, 51.2089767, -135.4302521, 135.4481506
42: -66.3184052, 38.0766449, -66.4215546, 38.0942612, -104.4126511, 104.4981995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1544
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
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1565
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
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 577
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
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 736
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

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7099685, upper bound: 75.7622553
time: 118.59 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7099686, upper bound: 75.7622553
time: 108.51 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -91.6727295, 66.0447083, -91.5727692, 65.8601379, -157.5328674, 157.6174622
1: -45.7724419, 56.5632210, -45.7223740, 56.1662598, -101.9386826, 102.2855911
2: -40.0940857, 57.6574707, -40.0418434, 57.2541199, -97.3482056, 97.6993103
3: -50.1455956, 59.8688736, -50.0934677, 59.2841568, -109.4297485, 109.9623337
4: -49.0352173, 73.7259216, -48.9403000, 73.2034073, -122.2386169, 122.6662216
5: -46.4310341, 58.6467400, -46.3526382, 58.1761627, -104.6071854, 104.9993744
6: -91.1626740, 43.8610306, -91.0789185, 43.8083992, -134.9710693, 134.9399414
7: -55.1200066, 57.0582886, -55.0327415, 56.7507668, -111.8707657, 112.0910339
8: -60.9106407, 83.3383179, -60.8522720, 82.8682709, -143.7789154, 144.1905823
9: -49.7454643, 63.8255653, -49.7068520, 63.6925964, -113.4380493, 113.5324097
10: -77.5941467, 72.3300018, -77.1785049, 72.1971283, -149.7912750, 149.5085144
11: -81.9633026, 37.7229614, -81.2264557, 37.6867981, -119.6501007, 118.9494171
12: -85.5819016, 51.5526123, -85.2284698, 51.4878883, -137.0697632, 136.7810822
13: -77.5921173, 81.2805405, -77.5931396, 80.8757095, -158.4678192, 158.8736725
14: -118.4114761, 55.8915215, -117.8475189, 55.8637810, -174.2752380, 173.7390442
15: -60.7048264, 63.6512337, -60.6588020, 63.4131088, -124.1179352, 124.3100204
16: -79.9088211, 54.9881020, -79.5561523, 54.9030571, -134.8118591, 134.5442505
17: -111.5817642, 47.9655914, -110.9160919, 47.9191971, -159.5009613, 158.8816833
18: -79.6374664, 54.4441528, -79.0521698, 54.4037704, -134.0412292, 133.4963074
19: -58.4437332, 36.1772079, -57.8154831, 36.1485901, -94.5923157, 93.9926834
20: -56.9428253, 39.8837280, -56.6149025, 39.8626556, -96.8054810, 96.4986191
21: -74.9966431, 41.6920815, -74.2823334, 41.6593094, -116.6559525, 115.9744110
22: -69.4122925, 44.1579056, -69.1165619, 44.1833496, -113.5956345, 113.2744675
23: -62.1357117, 46.7965698, -61.6098633, 46.7553253, -108.8910217, 108.4064255
24: -73.7890472, 46.1879501, -73.4899139, 46.1888390, -119.9778824, 119.6778641
25: -64.5839081, 47.6072502, -64.1901550, 47.5870743, -112.1709747, 111.7974091
26: -83.6299438, 62.0193596, -83.0364304, 61.9547920, -145.5847321, 145.0557861
27: -69.6754227, 45.9723320, -69.4499664, 45.9895554, -115.6649780, 115.4223022
28: -58.7321434, 48.8980713, -58.3882751, 48.8589706, -107.5911102, 107.2863464
29: -75.6506348, 42.3149109, -75.1649170, 42.3105087, -117.9611435, 117.4798279
30: -79.6904449, 47.9994354, -79.0620804, 47.9502411, -127.6406555, 127.0615082
31: -81.0008698, 47.9492493, -80.2870102, 47.9357376, -128.9366150, 128.2362671
32: -83.7467041, 42.8628998, -83.6893158, 42.7780075, -126.5246811, 126.5522079
33: -110.0013351, 52.6350212, -109.9337006, 52.3388672, -162.3402100, 162.5687256
34: -97.9328003, 28.7599602, -97.8749084, 28.7062569, -126.6390381, 126.6348724
35: -91.5918655, 40.0416832, -91.5768738, 39.8860054, -131.4778595, 131.6185608
36: -90.1392212, 45.7734299, -90.0894165, 45.6180267, -135.7572479, 135.8628540
37: -131.6855011, 40.5306396, -131.5303802, 40.4849396, -172.1704407, 172.0610199
38: -106.9434891, 49.9828377, -106.8594894, 49.7360420, -156.6795349, 156.8423309
39: -118.7785797, 57.5013046, -118.6785355, 57.3126831, -176.0912628, 176.1798248
40: -100.4150543, 35.5275993, -100.2852325, 35.3766060, -135.7916565, 135.8128357
41: -84.3657074, 51.2914543, -84.2744751, 51.1970673, -135.5627747, 135.5659332
42: -66.4737625, 38.2031746, -66.4161682, 38.0676804, -104.5414352, 104.6193314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 577
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.8292401, upper bound: 75.7385940
time: 97.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7099686, upper bound: 75.7385940
time: 91.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 191.94 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.8282968, upper bound: 75.8282968
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.8282968, upper bound: 75.8603185
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7818570, upper bound: 75.7296636
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7436136, upper bound: 75.7706865
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8725436
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7913681, upper bound: 75.9178818
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7399020, upper bound: 75.7813232
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7161960, upper bound: 75.8292401
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7099685, upper bound: 75.7622553
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7099686, upper bound: 75.7622553
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.8292401, upper bound: 75.7385940
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 191.94
Output dim: 4, lower bound: -75.7099686, upper bound: 75.7385940

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.2095337, 65.7256622, -156.9351959, 156.9351959
1: -45.4789619, 55.9934158, -45.4789619, 55.9934158, -101.4723587, 101.4723816
2: -39.7130814, 57.0314484, -39.7130814, 57.0314484, -96.7445221, 96.7445221
3: -49.7763100, 59.0561409, -49.7763100, 59.0561409, -108.8324432, 108.8324509
4: -48.5746574, 72.9390411, -48.5746574, 72.9390411, -121.5137024, 121.5137024
5: -45.9998627, 57.9758530, -45.9998627, 57.9758530, -103.9757156, 103.9757156
6: -90.7651367, 43.6791229, -90.7651367, 43.6791229, -134.4442444, 134.4442596
7: -54.6963768, 56.6240845, -54.6963768, 56.6240845, -111.3204498, 111.3204575
8: -60.4719582, 82.5659637, -60.4719582, 82.5659637, -143.0379181, 143.0379181
9: -49.3790245, 63.3509789, -49.3790245, 63.3509789, -112.7300034, 112.7300034
10: -76.4372025, 71.5561295, -76.4372025, 71.5561295, -147.9933319, 147.9933319
11: -80.5086060, 37.2519913, -80.5086060, 37.2519913, -117.7605972, 117.7605972
12: -84.6735001, 50.8577003, -84.6735001, 50.8577003, -135.5312042, 135.5312042
13: -77.4382019, 80.5325470, -77.4382019, 80.5325470, -157.9707336, 157.9707336
14: -117.1471558, 55.2790070, -117.1471558, 55.2790070, -172.4261475, 172.4261627
15: -60.2847214, 63.0727425, -60.2847214, 63.0727425, -123.3574524, 123.3574677
16: -78.9737396, 54.4944534, -78.9737396, 54.4944534, -133.4681702, 133.4681854
17: -110.5609436, 47.5792503, -110.5609436, 47.5792503, -158.1401978, 158.1401825
18: -78.7295837, 54.1633835, -78.7295837, 54.1633835, -132.8929443, 132.8929596
19: -57.5772057, 35.9203415, -57.5772057, 35.9203415, -93.4975433, 93.4975433
20: -56.2872849, 39.5818024, -56.2872849, 39.5818024, -95.8690872, 95.8690872
21: -73.8327789, 41.3040771, -73.8327789, 41.3040771, -115.1368561, 115.1368561
22: -68.8582230, 43.9032135, -68.8582230, 43.9032135, -112.7614365, 112.7614365
23: -61.4061775, 46.5395050, -61.4061775, 46.5395050, -107.9456787, 107.9456787
24: -73.2307129, 46.1248322, -73.2307129, 46.1248322, -119.3555450, 119.3555450
25: -64.0237274, 47.3331070, -64.0237274, 47.3331070, -111.3568344, 111.3568268
26: -82.7070160, 61.4796524, -82.7070160, 61.4796524, -144.1866608, 144.1866760
27: -69.0891266, 45.8833313, -69.0891266, 45.8833313, -114.9724579, 114.9724579
28: -58.2086182, 48.6852188, -58.2086182, 48.6852188, -106.8938370, 106.8938370
29: -74.9307251, 42.0965996, -74.9307251, 42.0965996, -117.0273132, 117.0273209
30: -78.8560181, 47.6862679, -78.8560181, 47.6862679, -126.5422821, 126.5422821
31: -79.9453430, 47.6296501, -79.9453430, 47.6296501, -127.5749969, 127.5749893
32: -83.3446808, 42.5247154, -83.3446808, 42.5247154, -125.8693848, 125.8693924
33: -109.5497742, 51.9672089, -109.5497742, 51.9672089, -161.5169678, 161.5169830
34: -97.5978851, 28.3786812, -97.5978851, 28.3786812, -125.9765625, 125.9765625
35: -91.3086853, 39.5517616, -91.3086853, 39.5517616, -130.8604431, 130.8604431
36: -89.8961334, 45.4528236, -89.8961334, 45.4528236, -135.3489532, 135.3489532
37: -131.2169189, 40.2779846, -131.2169189, 40.2779846, -171.4948883, 171.4949036
38: -106.5425110, 49.4991035, -106.5425110, 49.4991035, -156.0416107, 156.0416107
39: -118.3792419, 57.0401192, -118.3792419, 57.0401192, -175.4193573, 175.4193420
40: -99.9716492, 35.2111969, -99.9716492, 35.2111969, -135.1828308, 135.1828461
41: -84.0579147, 51.0648232, -84.0579147, 51.0648232, -135.1227417, 135.1227417
42: -66.1099396, 37.9253464, -66.1099396, 37.9253464, -104.0352783, 104.0352783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

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
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 705
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
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 673
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
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1718
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
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1463
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
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 901
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
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
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
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.6999874, upper bound: 75.7929676
time: 90.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7436133, upper bound: 75.7555161
time: 104.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.4338989, 65.8267288, -157.0362549, 157.1595612
1: -45.4789619, 55.9934158, -45.6207848, 56.1456070, -101.6245728, 101.6141968
2: -39.7130814, 57.0314484, -39.8771133, 57.2357330, -96.9488068, 96.9085617
3: -49.7763100, 59.0561409, -49.9112701, 59.2532692, -109.0295792, 108.9674072
4: -48.5746574, 72.9390411, -48.7564049, 73.1786346, -121.7532959, 121.6954498
5: -45.9998627, 57.9758530, -46.1513596, 58.1457367, -104.1455994, 104.1272049
6: -90.7651367, 43.6791229, -91.0247345, 43.7854691, -134.5505981, 134.7038574
7: -54.6963768, 56.6240845, -54.8932571, 56.7288094, -111.4251862, 111.5173340
8: -60.4719582, 82.5659637, -60.6793213, 82.8389130, -143.3108673, 143.2452850
9: -49.3790245, 63.3509789, -49.6666260, 63.5584526, -112.9374695, 113.0176086
10: -76.4372025, 71.5561295, -77.1121521, 71.9176483, -148.3548431, 148.6682739
11: -80.5086060, 37.2519913, -81.1769180, 37.4926949, -118.0012970, 118.4289093
12: -84.6735001, 50.8577003, -85.1918182, 51.1994934, -135.8729858, 136.0495148
13: -77.4382019, 80.5325470, -77.5595093, 80.7984161, -158.2366180, 158.0920410
14: -117.1471558, 55.2790070, -117.7800751, 55.6163406, -172.7634888, 173.0590820
15: -60.2847214, 63.0727425, -60.5569153, 63.3658943, -123.6506042, 123.6296539
16: -78.9737396, 54.4944534, -79.4862671, 54.7474098, -133.7211456, 133.9807129
17: -110.5609436, 47.5792503, -110.8771286, 47.8085785, -158.3695068, 158.4563599
18: -78.7295837, 54.1633835, -79.0050049, 54.3086586, -133.0382385, 133.1683807
19: -57.5772057, 35.9203415, -57.7675934, 36.0266876, -93.6038818, 93.6879349
20: -56.2872849, 39.5818024, -56.5647888, 39.7293396, -96.0166245, 96.1465912
21: -73.8327789, 41.3040771, -74.2364197, 41.4875984, -115.3203735, 115.5404892
22: -68.8582230, 43.9032135, -69.0749054, 44.1135368, -112.9717484, 112.9781189
23: -61.4061775, 46.5395050, -61.5722313, 46.6534843, -108.0596619, 108.1117401
24: -73.2307129, 46.1248322, -73.4411926, 46.1718292, -119.4025269, 119.5660248
25: -64.0237274, 47.3331070, -64.1577988, 47.4803467, -111.5040741, 111.4909058
26: -82.7070160, 61.4796524, -82.9858704, 61.7168350, -144.4238586, 144.4655151
27: -69.0891266, 45.8833313, -69.3677979, 45.9666443, -115.0557709, 115.2511292
28: -58.2086182, 48.6852188, -58.3462563, 48.7567825, -106.9653931, 107.0314713
29: -74.9307251, 42.0965996, -75.1340790, 42.2264862, -117.1572113, 117.2306824
30: -78.8560181, 47.6862679, -79.0244064, 47.8249283, -126.6809311, 126.7106705
31: -79.9453430, 47.6296501, -80.2249451, 47.7910309, -127.7363586, 127.8545990
32: -83.3446808, 42.5247154, -83.6553040, 42.6757812, -126.0204620, 126.1800079
33: -109.5497742, 51.9672089, -109.8073120, 52.2885361, -161.8383179, 161.7745056
34: -97.5978851, 28.3786812, -97.7927094, 28.6629028, -126.2607880, 126.1713791
35: -91.3086853, 39.5517616, -91.5038528, 39.8465118, -131.1551819, 131.0556183
36: -89.8961334, 45.4528236, -90.0530090, 45.5732002, -135.4693298, 135.5058289
37: -131.2169189, 40.2779846, -131.4750671, 40.4192009, -171.6361237, 171.7530518
38: -106.5425110, 49.4991035, -106.7706680, 49.6926270, -156.2351227, 156.2697754
39: -118.3792419, 57.0401192, -118.6032639, 57.2369156, -175.6161346, 175.6433716
40: -99.9716492, 35.2111969, -100.2222214, 35.3467407, -135.3183899, 135.4334106
41: -84.0579147, 51.0648232, -84.2253418, 51.1602936, -135.2182007, 135.2901611
42: -66.1099396, 37.9253464, -66.3718262, 38.0065536, -104.1164856, 104.2971649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

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
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 705
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
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 673
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
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1718
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
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1463
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
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 901
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
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
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
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.6999874, upper bound: 75.8117902
time: 102.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7436133, upper bound: 75.7706868
time: 105.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 211.23 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 211.23
Output dim: 4, lower bound: -75.6999874, upper bound: 75.7929676
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 211.23
Output dim: 4, lower bound: -75.7436133, upper bound: 75.7555161
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 211.23
Output dim: 4, lower bound: -75.6999874, upper bound: 75.8117902
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 211.23
Output dim: 4, lower bound: -75.7436133, upper bound: 75.7706868
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7818570, upper bound: 75.7296636
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7436136, upper bound: 75.7706865
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7913681, upper bound: 75.8725436
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7913681, upper bound: 75.9178818
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7399020, upper bound: 75.7813232
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7161960, upper bound: 75.8292401
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7980559, upper bound: 75.7913682
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7099685, upper bound: 75.7622553
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7099686, upper bound: 75.7622553
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.8292401, upper bound: 75.7385940
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 211.23
Output dim: 4, lower bound: -75.7099686, upper bound: 75.7385940
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=121.93731689453125
rel_dist={4: [-75.94959621478742, 75.94959622088305]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

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
Output dim: 4, lower bound: -74.6161867, upper bound: 74.7007251
time: 117.15 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.6161867, upper bound: 74.7007251
time: 78.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 195.76 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 195.76
Output dim: 4, lower bound: -74.6161867, upper bound: 74.7007251
IS_A2, status: Status.UNKNOWN, split count: 1, time: 195.76
Output dim: 4, lower bound: -74.6161867, upper bound: 74.7007251

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -91.3213043, 65.7857513, -91.4565353, 65.8212051, -157.1425018, 157.2422638
1: -45.5418701, 56.0182419, -45.6386490, 56.0393600, -101.5812302, 101.6568832
2: -39.8204613, 57.0575562, -39.9698029, 57.0782318, -96.8986969, 97.0273590
3: -49.8970680, 59.1013107, -50.0600510, 59.1343536, -109.0314178, 109.1613617
4: -48.7142906, 72.9826508, -48.8813705, 73.0092773, -121.7235565, 121.8640213
5: -46.1217842, 58.0246353, -46.2966080, 58.0563354, -104.1781082, 104.3212357
6: -90.8448639, 43.7711296, -90.8997269, 43.8291473, -134.6740112, 134.6708374
7: -54.7714386, 56.6660614, -54.9015808, 56.6890488, -111.4604874, 111.5676422
8: -60.6149902, 82.6140213, -60.7715225, 82.6461029, -143.2610626, 143.3855438
9: -49.4227333, 63.5123978, -49.4676552, 63.6342239, -113.0569611, 112.9800491
10: -76.5075378, 71.8824463, -76.5719452, 72.1379852, -148.6455231, 148.4543915
11: -80.5862885, 37.4702682, -80.6386795, 37.6514359, -118.2377243, 118.1089478
12: -84.7255707, 51.1341171, -84.7631073, 51.3915138, -136.1170654, 135.8972168
13: -77.4858856, 80.6926575, -77.5523834, 80.7665405, -158.2524109, 158.2450409
14: -117.2267990, 55.5879097, -117.2960815, 55.8154831, -173.0422668, 172.8839722
15: -60.4579048, 63.1360359, -60.5831757, 63.1840477, -123.6419525, 123.7192078
16: -79.0574265, 54.7088737, -79.1321335, 54.8747711, -133.9321747, 133.8410034
17: -110.6097488, 47.7281609, -110.6508026, 47.8381271, -158.4478760, 158.3789673
18: -78.8070526, 54.2625046, -78.8566589, 54.3662872, -133.1733398, 133.1191711
19: -57.6342239, 36.0015602, -57.6808701, 36.1162186, -93.7504349, 93.6824341
20: -56.3461571, 39.7109489, -56.3946381, 39.8364563, -96.1826172, 96.1055756
21: -73.8945923, 41.4410095, -73.9409561, 41.5996780, -115.4942703, 115.3819656
22: -68.9924240, 43.9675217, -69.0337067, 44.0450058, -113.0374146, 113.0012207
23: -61.4508209, 46.6096725, -61.4882698, 46.7080765, -108.1588898, 108.0979309
24: -73.3691940, 46.1511002, -73.4233780, 46.1884613, -119.5576477, 119.5744781
25: -64.0892487, 47.4152641, -64.1248474, 47.5176010, -111.6068268, 111.5401077
26: -82.7797394, 61.6532631, -82.8318481, 61.8675652, -144.6473083, 144.4851074
27: -69.2690811, 45.9126091, -69.3488007, 45.9467621, -115.2158432, 115.2614059
28: -58.2973595, 48.7280655, -58.3379593, 48.8253021, -107.1226654, 107.0660172
29: -75.0226898, 42.1504745, -75.0553741, 42.2364311, -117.2591248, 117.2058411
30: -78.9212570, 47.7777481, -78.9628906, 47.8983917, -126.8196411, 126.7406311
31: -80.0189209, 47.7652168, -80.0796967, 47.9031143, -127.9220352, 127.8449097
32: -83.4178314, 42.6529427, -83.4609375, 42.7531815, -126.1710052, 126.1138611
33: -109.7412338, 52.0245743, -109.8631134, 52.0733871, -161.8146210, 161.8876953
34: -97.7315979, 28.4245987, -97.8119888, 28.4693451, -126.2009125, 126.2365799
35: -91.4567719, 39.5986328, -91.5364380, 39.6363716, -131.0931396, 131.1350708
36: -89.9831314, 45.4956589, -90.0310440, 45.5401306, -135.5232544, 135.5267029
37: -131.3728943, 40.3259125, -131.4377747, 40.4038277, -171.7767181, 171.7636871
38: -106.6347885, 49.6071091, -106.7282944, 49.6506882, -156.2854767, 156.3354034
39: -118.4963913, 57.1340942, -118.5775452, 57.2044945, -175.7008667, 175.7116394
40: -100.0724792, 35.2577820, -100.1374512, 35.2908325, -135.3633118, 135.3952332
41: -84.1393890, 51.1012497, -84.1945953, 51.1498680, -135.2892609, 135.2958374
42: -66.1784515, 38.0395699, -66.2214661, 38.1255341, -104.3039856, 104.2610321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1571
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 917
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
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 966
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
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1625
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
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6416266
time: 77.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6819221
time: 137.16 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -91.5572815, 65.9842834, -91.4768372, 65.8264008, -157.3836670, 157.4611206
1: -45.7041969, 56.2767944, -45.6542892, 56.0422745, -101.7464676, 101.9310837
2: -40.0440826, 57.3512192, -39.9957352, 57.0803909, -97.1244659, 97.3469543
3: -50.1253052, 59.4354057, -50.0861626, 59.1383171, -109.2636185, 109.5215607
4: -48.9753952, 73.3160248, -48.9105835, 73.0115967, -121.9869843, 122.2266083
5: -46.3873367, 58.3225212, -46.3305511, 58.0602951, -104.4476318, 104.6530685
6: -91.0342331, 43.8610458, -90.9067307, 43.8208542, -134.8550873, 134.7677765
7: -55.0119286, 56.8803864, -54.9222450, 56.6916046, -111.7035370, 111.8026276
8: -60.8534088, 82.9740524, -60.7982025, 82.6503906, -143.5037994, 143.7722473
9: -49.6685791, 63.7269135, -49.4730797, 63.6554108, -113.3239899, 113.1999893
10: -77.0238342, 72.2536621, -76.5808105, 72.1819229, -149.2057495, 148.8344727
11: -81.0911407, 37.7204285, -80.6462784, 37.6814728, -118.7726135, 118.3667068
12: -85.1889343, 51.4995117, -84.7677689, 51.4370728, -136.6260071, 136.2672729
13: -77.5886230, 80.8915405, -77.5501862, 80.7776642, -158.3662872, 158.4417267
14: -117.6962051, 55.9012146, -117.3057632, 55.8550415, -173.5512390, 173.2069702
15: -60.6601868, 63.3568382, -60.5852661, 63.1912193, -123.8514023, 123.9421082
16: -79.4302521, 54.9584122, -79.1413193, 54.8958130, -134.3260651, 134.0997314
17: -110.9078140, 47.9288902, -110.6564407, 47.8534203, -158.7612305, 158.5853271
18: -79.0908661, 54.4297676, -78.8641434, 54.3822708, -133.4731293, 133.2939148
19: -57.9276581, 36.1629868, -57.6878510, 36.1346436, -94.0623016, 93.8508377
20: -56.6052322, 39.8797951, -56.4021034, 39.8566284, -96.4618607, 96.2818909
21: -74.3057709, 41.6711502, -73.9469604, 41.6268463, -115.9326172, 115.6181107
22: -69.1615906, 44.1064987, -69.0387650, 44.0479507, -113.2095413, 113.1452560
23: -61.6743546, 46.7657280, -61.4933968, 46.7246437, -108.3989868, 108.2591248
24: -73.4954834, 46.2025146, -73.4264984, 46.1864395, -119.6819229, 119.6290131
25: -64.2450790, 47.5862198, -64.1287689, 47.5343590, -111.7794342, 111.7149887
26: -83.1230164, 61.9699249, -82.8382034, 61.9043427, -145.0273438, 144.8081055
27: -69.4592133, 45.9666443, -69.3605804, 45.9471436, -115.4063568, 115.3272247
28: -58.4603424, 48.8746986, -58.3440323, 48.8413315, -107.3016739, 107.2187271
29: -75.2252655, 42.2832603, -75.0584869, 42.2441330, -117.4693985, 117.3417511
30: -79.1736145, 47.9719620, -78.9677734, 47.9181862, -127.0917892, 126.9397354
31: -80.3591156, 47.9559555, -80.0894165, 47.9247360, -128.2838440, 128.0453796
32: -83.6677551, 42.8102150, -83.4662857, 42.7692299, -126.4369812, 126.2765045
33: -109.9391403, 52.2600403, -109.8821030, 52.0808868, -162.0200195, 162.1421356
34: -97.8776474, 28.5842056, -97.8240433, 28.4757919, -126.3534241, 126.4082413
35: -91.5763474, 39.7871742, -91.5456390, 39.6428909, -131.2192383, 131.3328094
36: -90.1005859, 45.6061096, -90.0348358, 45.5429001, -135.6434937, 135.6409302
37: -131.5559387, 40.4748306, -131.4433289, 40.4119606, -171.9678955, 171.9181519
38: -106.8374786, 49.7533455, -106.7394485, 49.6555939, -156.4930725, 156.4927979
39: -118.6868591, 57.3069305, -118.5880966, 57.2143478, -175.9011993, 175.8950195
40: -100.2728806, 35.3652458, -100.1462936, 35.2900314, -135.5628967, 135.5115356
41: -84.2885742, 51.1965332, -84.2017212, 51.1505508, -135.4391174, 135.3982544
42: -66.3551025, 38.1627693, -66.2265167, 38.1154900, -104.4705811, 104.3892822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1571
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
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 995
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
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1020
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
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 966
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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6416266
time: 115.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6819221
time: 101.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 218.84 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 218.84
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6416266
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 218.84
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6819221
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 218.84
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6416266
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 218.84
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6819221

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -91.3122635, 65.7809906, -91.3442154, 65.7612381, -157.0735016, 157.1252136
1: -45.5368271, 56.0162811, -45.5757980, 56.0146675, -101.5514984, 101.5920715
2: -39.8118668, 57.0554810, -39.8625755, 57.0522690, -96.8641357, 96.9180527
3: -49.8873405, 59.0977249, -49.9391632, 59.0894203, -108.9767456, 109.0368881
4: -48.7031364, 72.9792328, -48.7419014, 72.9658813, -121.6689911, 121.7211227
5: -46.1120911, 58.0208092, -46.1711044, 58.0077209, -104.1198120, 104.1919098
6: -90.8385620, 43.7634506, -90.8203888, 43.7371330, -134.5756989, 134.5838318
7: -54.7654381, 56.6627121, -54.8259506, 56.6470795, -111.4125061, 111.4886627
8: -60.6035767, 82.6102066, -60.6285019, 82.5982971, -143.2018280, 143.2387085
9: -49.4192505, 63.4993820, -49.4241219, 63.4728088, -112.8920593, 112.9235077
10: -76.5019836, 71.8565063, -76.5018768, 71.8117523, -148.3137207, 148.3583679
11: -80.5801926, 37.4529877, -80.5612640, 37.4332199, -118.0134125, 118.0142517
12: -84.7214508, 51.1122360, -84.7113342, 51.1151428, -135.8365936, 135.8235779
13: -77.4820709, 80.6800079, -77.5047150, 80.6067505, -158.0888062, 158.1847229
14: -117.2203903, 55.5634651, -117.2165604, 55.5062904, -172.7266846, 172.7800140
15: -60.4441605, 63.1310349, -60.4099274, 63.1210670, -123.5652313, 123.5409622
16: -79.0507736, 54.6917267, -79.0490799, 54.6604233, -133.7111969, 133.7408142
17: -110.6057816, 47.7160645, -110.6021347, 47.6870346, -158.2928162, 158.3182068
18: -78.8008347, 54.2546387, -78.7789154, 54.2670784, -133.0679169, 133.0335541
19: -57.6296692, 35.9949646, -57.6239624, 36.0351486, -93.6648102, 93.6189270
20: -56.3414955, 39.7007294, -56.3359413, 39.7074127, -96.0488815, 96.0366669
21: -73.8896942, 41.4302216, -73.8793030, 41.4628830, -115.3525772, 115.3095245
22: -68.9817352, 43.9624634, -68.8994293, 43.9809761, -112.9627075, 112.8618927
23: -61.4472656, 46.6040039, -61.4437027, 46.6379242, -108.0851898, 108.0476990
24: -73.3581696, 46.1490173, -73.2848587, 46.1619873, -119.5201569, 119.4338760
25: -64.0840454, 47.4087448, -64.0593414, 47.4352722, -111.5193100, 111.4680862
26: -82.7739639, 61.6393738, -82.7593384, 61.6923752, -144.4663391, 144.3987122
27: -69.2548065, 45.9103088, -69.1685257, 45.9175262, -115.1723175, 115.0788345
28: -58.2903709, 48.7246475, -58.2492447, 48.7820435, -107.0724030, 106.9738846
29: -75.0153656, 42.1462097, -74.9634247, 42.1826973, -117.1980591, 117.1096115
30: -78.9160919, 47.7702942, -78.8979340, 47.8069153, -126.7230072, 126.6682281
31: -80.0130844, 47.7545166, -80.0063477, 47.7676735, -127.7807465, 127.7608643
32: -83.4120102, 42.6427269, -83.3878479, 42.6250763, -126.0370865, 126.0305786
33: -109.7261505, 52.0200157, -109.6715927, 52.0160866, -161.7422180, 161.6916046
34: -97.7208786, 28.4209213, -97.6780701, 28.4234734, -126.1443481, 126.0989914
35: -91.4450684, 39.5950012, -91.3882751, 39.5897446, -131.0348053, 130.9832764
36: -89.9762573, 45.4921570, -89.9438934, 45.4960632, -135.4723206, 135.4360504
37: -131.3605499, 40.3220596, -131.2815857, 40.3558578, -171.7163849, 171.6036377
38: -106.6274338, 49.5981979, -106.6358871, 49.5420074, -156.1694336, 156.2340698
39: -118.4870682, 57.1266251, -118.4603119, 57.1100845, -175.5971375, 175.5869446
40: -100.0644760, 35.2538719, -100.0369568, 35.2432899, -135.3077698, 135.2908325
41: -84.1329041, 51.0982323, -84.1132507, 51.1132736, -135.2461853, 135.2114868
42: -66.1730499, 38.0303230, -66.1534119, 38.0112839, -104.1843338, 104.1837311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1547
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
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1022
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
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1555
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 942
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
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
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
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1552
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
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1467
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
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5687592, upper bound: 74.6416266
time: 92.19 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6416266
time: 91.26 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -91.3047180, 65.7669678, -91.5687408, 65.8625946, -157.1673126, 157.3356934
1: -45.5331497, 56.0154877, -45.7169609, 56.1666794, -101.6998291, 101.7324524
2: -39.8066101, 57.0523872, -40.0259094, 57.2561989, -97.0628052, 97.0782928
3: -49.8746605, 59.0941048, -50.0751953, 59.2861023, -109.1607666, 109.1692963
4: -48.6828003, 72.9741669, -48.9229889, 73.2050781, -121.8878784, 121.8971481
5: -46.0980721, 58.0183296, -46.3294525, 58.1773911, -104.2754517, 104.3477783
6: -90.8309555, 43.7411919, -91.0797882, 43.8433151, -134.6742706, 134.8209839
7: -54.7572823, 56.6591415, -55.0224533, 56.7519455, -111.5092316, 111.6815948
8: -60.5991745, 82.6073914, -60.8351669, 82.8705521, -143.4697266, 143.4425659
9: -49.4152756, 63.4965897, -49.7112427, 63.6796722, -113.0949478, 113.2078323
10: -76.4984741, 71.8522186, -77.1761017, 72.1721420, -148.6706238, 149.0283203
11: -80.5775299, 37.4497719, -81.2290421, 37.6734848, -118.2510147, 118.6788177
12: -84.7169342, 51.1112442, -85.2292328, 51.4562225, -136.1731415, 136.3404846
13: -77.4750061, 80.6707001, -77.6259766, 80.8753281, -158.3503418, 158.2966614
14: -117.2118835, 55.5634689, -117.8489609, 55.8435555, -173.0554352, 173.4124298
15: -60.4420128, 63.1273994, -60.6819038, 63.4129333, -123.8549500, 123.8093033
16: -79.0434113, 54.6873398, -79.5600433, 54.9125633, -133.9559784, 134.2473755
17: -110.6003876, 47.7030182, -110.9180679, 47.9189186, -158.5192871, 158.6210938
18: -78.7898560, 54.2453995, -79.0548401, 54.4122772, -133.2021179, 133.3002319
19: -57.6251945, 35.9884262, -57.8144417, 36.1409569, -93.7661514, 93.8028641
20: -56.3400116, 39.6984291, -56.6130638, 39.8545227, -96.1945343, 96.3114929
21: -73.8850021, 41.4292870, -74.2826080, 41.6457558, -115.5307465, 115.7118912
22: -68.9644012, 43.9577103, -69.1165771, 44.1905785, -113.1549835, 113.0742874
23: -61.4444427, 46.6014481, -61.6098404, 46.7514496, -108.1958923, 108.2112885
24: -73.3535767, 46.1421509, -73.4954681, 46.2090759, -119.5626373, 119.6376190
25: -64.0770569, 47.4059982, -64.1938095, 47.5824814, -111.6595306, 111.5998077
26: -82.7699127, 61.6298065, -83.0382156, 61.9307175, -144.7006226, 144.6680298
27: -69.2519989, 45.9074020, -69.4470749, 46.0006256, -115.2526245, 115.3544769
28: -58.2810287, 48.7175598, -58.3871651, 48.8535843, -107.1345978, 107.1047211
29: -75.0091858, 42.1412659, -75.1670837, 42.3110809, -117.3202667, 117.3083496
30: -78.9133224, 47.7616882, -79.0665359, 47.9454193, -126.8587341, 126.8282242
31: -80.0101624, 47.7511482, -80.2853851, 47.9286728, -127.9388351, 128.0365295
32: -83.4069977, 42.6386757, -83.6980743, 42.7757263, -126.1827240, 126.3367462
33: -109.7230301, 52.0175552, -109.9289627, 52.3369980, -162.0600281, 161.9465179
34: -97.7179642, 28.4180756, -97.8729172, 28.7072792, -126.4252319, 126.2909927
35: -91.4420471, 39.5937881, -91.5833435, 39.8834801, -131.3255310, 131.1771240
36: -89.9682312, 45.4868546, -90.1008530, 45.6177292, -135.5859680, 135.5877075
37: -131.3484802, 40.3200226, -131.5399475, 40.4960938, -171.8445740, 171.8599701
38: -106.6174774, 49.5871582, -106.8631744, 49.7362900, -156.3537598, 156.4503326
39: -118.4722061, 57.1268730, -118.6855927, 57.3067245, -175.7789307, 175.8124695
40: -100.0593948, 35.2468147, -100.2871246, 35.3803291, -135.4397278, 135.5339355
41: -84.1238556, 51.0943565, -84.2806015, 51.2076874, -135.3315430, 135.3749390
42: -66.1676178, 37.9922905, -66.4153137, 38.0940666, -104.2616806, 104.4076080

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
type: A, layer: 1, pos: 1687
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 523
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
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1368
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
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4919848, upper bound: 74.6331884
time: 98.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5198832, upper bound: 74.6031303
time: 105.07 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -91.5482254, 65.9795990, -91.3644485, 65.7664185, -157.3146362, 157.3440552
1: -45.6991119, 56.2748337, -45.5914078, 56.0175934, -101.7167053, 101.8662415
2: -40.0354996, 57.3491783, -39.8885117, 57.0544319, -97.0899277, 97.2376862
3: -50.1154671, 59.4318657, -49.9652939, 59.0934334, -109.2088928, 109.3971558
4: -48.9642525, 73.3126373, -48.7711105, 72.9682159, -121.9324646, 122.0837402
5: -46.3772278, 58.3187332, -46.2035370, 58.0116959, -104.3889236, 104.5222626
6: -91.0279465, 43.8534889, -90.8274078, 43.7288589, -134.7568054, 134.6808929
7: -55.0057716, 56.8771324, -54.8464661, 56.6496048, -111.6553802, 111.7236023
8: -60.8419838, 82.9702682, -60.6552048, 82.6026154, -143.4445953, 143.6254730
9: -49.6651421, 63.7139359, -49.4295883, 63.4939766, -113.1591187, 113.1435242
10: -77.0182800, 72.2277374, -76.5107880, 71.8557281, -148.8739929, 148.7385254
11: -81.0850449, 37.7031746, -80.5689316, 37.4632912, -118.5483322, 118.2721024
12: -85.1848602, 51.4777184, -84.7160034, 51.1607056, -136.3455658, 136.1937103
13: -77.5848465, 80.8789520, -77.5024109, 80.6179199, -158.2027588, 158.3813477
14: -117.6899109, 55.8768311, -117.2262115, 55.5458832, -173.2357788, 173.1030426
15: -60.6464119, 63.3518829, -60.4118919, 63.1283455, -123.7747574, 123.7637634
16: -79.4236526, 54.9413261, -79.0583191, 54.6814766, -134.1051331, 133.9996338
17: -110.9039154, 47.9168015, -110.6077652, 47.7022552, -158.6061707, 158.5245667
18: -79.0848083, 54.4219093, -78.7862778, 54.2829971, -133.3677979, 133.2081909
19: -57.9231796, 36.1564636, -57.6309433, 36.0536118, -93.9767914, 93.7874069
20: -56.6005707, 39.8695908, -56.3434258, 39.7275276, -96.3280869, 96.2130127
21: -74.3009033, 41.6604156, -73.8853455, 41.4900475, -115.7909546, 115.5457611
22: -69.1509171, 44.1013947, -68.9044952, 43.9841843, -113.1351013, 113.0058823
23: -61.6708717, 46.7600555, -61.4488258, 46.6545258, -108.3253860, 108.2088776
24: -73.4844971, 46.2003174, -73.2879868, 46.1596832, -119.6441650, 119.4883041
25: -64.2400360, 47.5797043, -64.0632858, 47.4520111, -111.6920395, 111.6429901
26: -83.1173019, 61.9557724, -82.7657013, 61.7291107, -144.8464050, 144.7214661
27: -69.4449463, 45.9643021, -69.1801987, 45.9178696, -115.3628159, 115.1445007
28: -58.4533501, 48.8712006, -58.2553368, 48.7980919, -107.2514343, 107.1265259
29: -75.2181702, 42.2789154, -74.9664917, 42.1904984, -117.4086685, 117.2454071
30: -79.1685333, 47.9644928, -78.9028320, 47.8267555, -126.9952850, 126.8673172
31: -80.3533630, 47.9452934, -80.0160675, 47.7892914, -128.1426544, 127.9613647
32: -83.6620026, 42.8000870, -83.3932266, 42.6410751, -126.3030777, 126.1932983
33: -109.9240265, 52.2555962, -109.6905594, 52.0235901, -161.9476013, 161.9461365
34: -97.8669357, 28.5805550, -97.6900482, 28.4299545, -126.2968903, 126.2705994
35: -91.5646210, 39.7834778, -91.3974457, 39.5963097, -131.1609344, 131.1809235
36: -90.0936813, 45.6025391, -89.9476547, 45.4985085, -135.5921936, 135.5502014
37: -131.5436707, 40.4709778, -131.2871857, 40.3640671, -171.9077148, 171.7581635
38: -106.8300858, 49.7445831, -106.6468735, 49.5467796, -156.3768616, 156.3914490
39: -118.6773834, 57.2993469, -118.4695435, 57.1190186, -175.7964020, 175.7688904
40: -100.2649612, 35.3613777, -100.0458755, 35.2423248, -135.5072784, 135.4072418
41: -84.2821350, 51.1935806, -84.1203766, 51.1139908, -135.3961182, 135.3139648
42: -66.3497238, 38.1535873, -66.1585236, 38.0012512, -104.3509750, 104.3121109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
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
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
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
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
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
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1476
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
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5687592, upper bound: 74.6416266
time: 134.05 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6416266
time: 116.05 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -91.5401611, 65.9648285, -91.5887833, 65.8679276, -157.4080811, 157.5536194
1: -45.6951408, 56.2741127, -45.7321587, 56.1696243, -101.8647614, 102.0062714
2: -40.0317993, 57.3460007, -40.0514450, 57.2583466, -97.2901459, 97.3974457
3: -50.1047211, 59.4283066, -50.1055908, 59.2901154, -109.3948364, 109.5338974
4: -48.9433746, 73.3075562, -48.9518051, 73.2076874, -122.1510620, 122.2593613
5: -46.3668785, 58.3161621, -46.3634453, 58.1812553, -104.5481262, 104.6796112
6: -91.0202332, 43.8304176, -91.0870209, 43.8350487, -134.8552856, 134.9174347
7: -54.9988289, 56.8732185, -55.0419273, 56.7545624, -111.7533875, 111.9151459
8: -60.8370972, 82.9674759, -60.8615723, 82.8747253, -143.7118073, 143.8290405
9: -49.6610031, 63.7106171, -49.7168159, 63.7004929, -113.3614960, 113.4274292
10: -77.0146484, 72.2226715, -77.1849060, 72.2151794, -149.2298279, 149.4075775
11: -81.0823593, 37.6995926, -81.2365417, 37.7034760, -118.7858276, 118.9361343
12: -85.1802902, 51.4761887, -85.2340927, 51.5014343, -136.6817169, 136.7102814
13: -77.5776367, 80.8696442, -77.6231232, 80.8868713, -158.4645081, 158.4927673
14: -117.6803513, 55.8767776, -117.8586426, 55.8830185, -173.5633545, 173.7354126
15: -60.6440277, 63.3482819, -60.6839142, 63.4199524, -124.0639801, 124.0321960
16: -79.4161377, 54.9361420, -79.5691757, 54.9332314, -134.3493652, 134.5053101
17: -110.8980026, 47.9036865, -110.9235535, 47.9346008, -158.8326111, 158.8272400
18: -79.0728836, 54.4133568, -79.0621643, 54.4291534, -133.5020447, 133.4755249
19: -57.9182892, 36.1493034, -57.8213577, 36.1591263, -94.0774078, 93.9706573
20: -56.5991173, 39.8669815, -56.6205521, 39.8743515, -96.4734650, 96.4875336
21: -74.2960663, 41.6590042, -74.2886353, 41.6725464, -115.9686127, 115.9476395
22: -69.1337128, 44.0963669, -69.1218414, 44.1939316, -113.3276443, 113.2182083
23: -61.6680107, 46.7570801, -61.6150169, 46.7677155, -108.4357224, 108.3721008
24: -73.4796600, 46.1939087, -73.4987106, 46.2079086, -119.6875610, 119.6926193
25: -64.2317963, 47.5768280, -64.1977386, 47.5991402, -111.8309326, 111.7745667
26: -83.1131134, 61.9489212, -83.0448456, 61.9720612, -145.0851746, 144.9937744
27: -69.4416733, 45.9613113, -69.4585571, 46.0009308, -115.4426041, 115.4198685
28: -58.4439087, 48.8642197, -58.3932152, 48.8695107, -107.3134155, 107.2574234
29: -75.2098083, 42.2737732, -75.1703491, 42.3188019, -117.5286102, 117.4441223
30: -79.1656342, 47.9551010, -79.0713959, 47.9652786, -127.1309128, 127.0264969
31: -80.3504486, 47.9415359, -80.2949829, 47.9502945, -128.3007507, 128.2365112
32: -83.6569061, 42.7953568, -83.7035294, 42.7914314, -126.4483337, 126.4988785
33: -109.9207687, 52.2522812, -109.9478836, 52.3443680, -162.2651062, 162.2001648
34: -97.8638229, 28.5770874, -97.8847656, 28.7134552, -126.5772552, 126.4618530
35: -91.5614624, 39.7818527, -91.5923767, 39.8897209, -131.4511871, 131.3742371
36: -90.0855560, 45.5978241, -90.1048126, 45.6210785, -135.7066345, 135.7026367
37: -131.5314789, 40.4685211, -131.5457611, 40.5037193, -172.0352020, 172.0142822
38: -106.8195877, 49.7333298, -106.8725281, 49.7414856, -156.5610657, 156.6058502
39: -118.6638489, 57.3004799, -118.6960297, 57.3167953, -175.9806519, 175.9965057
40: -100.2600250, 35.3539085, -100.2960663, 35.3797379, -135.6397400, 135.6499786
41: -84.2730560, 51.1888657, -84.2877808, 51.2070999, -135.4801483, 135.4766541
42: -66.3441849, 38.1167908, -66.4207611, 38.0900002, -104.4341812, 104.5375366

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
type: A, layer: 1, pos: 1687
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
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
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
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1515
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
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1657
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
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1476
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
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 955

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4919848, upper bound: 74.6331884
time: 231.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.6031299, upper bound: 74.6031303
time: 123.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 357.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 357.73
Output dim: 4, lower bound: -74.5687592, upper bound: 74.6416266
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 357.73
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6416266
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 357.73
Output dim: 4, lower bound: -74.4919848, upper bound: 74.6331884
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 357.73
Output dim: 4, lower bound: -74.5198832, upper bound: 74.6031303
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 357.73
Output dim: 4, lower bound: -74.5687592, upper bound: 74.6416266
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 357.73
Output dim: 4, lower bound: -74.5896907, upper bound: 74.6416266
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 357.73
Output dim: 4, lower bound: -74.4919848, upper bound: 74.6331884
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 357.73
Output dim: 4, lower bound: -74.6031299, upper bound: 74.6031303

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -91.2095337, 65.7256622, -91.3442154, 65.7612381, -156.9707642, 157.0698853
1: -45.4789619, 55.9934158, -45.5757980, 56.0146675, -101.4936142, 101.5692139
2: -39.7130814, 57.0314484, -39.8625755, 57.0522690, -96.7653503, 96.8940201
3: -49.7763100, 59.0561409, -49.9391632, 59.0894203, -108.8657303, 108.9953003
4: -48.5746574, 72.9390411, -48.7419014, 72.9658813, -121.5405426, 121.6809387
5: -45.9998627, 57.9758530, -46.1711044, 58.0077209, -104.0075684, 104.1469574
6: -90.7651367, 43.6791229, -90.8203888, 43.7371330, -134.5022583, 134.4995117
7: -54.6963768, 56.6240845, -54.8259506, 56.6470795, -111.3434448, 111.4500351
8: -60.4719582, 82.5659637, -60.6285019, 82.5982971, -143.0702362, 143.1944580
9: -49.3790245, 63.3509789, -49.4241219, 63.4728088, -112.8518372, 112.7751007
10: -76.4372025, 71.5561295, -76.5018768, 71.8117523, -148.2489319, 148.0580139
11: -80.5086060, 37.2519913, -80.5612640, 37.4332199, -117.9418259, 117.8132477
12: -84.6735001, 50.8577003, -84.7113342, 51.1151428, -135.7886353, 135.5690308
13: -77.4382019, 80.5325470, -77.5047150, 80.6067505, -158.0449524, 158.0372620
14: -117.1471558, 55.2790070, -117.2165604, 55.5062904, -172.6534424, 172.4955597
15: -60.2847214, 63.0727425, -60.4099274, 63.1210670, -123.4057922, 123.4826660
16: -78.9737396, 54.4944534, -79.0490799, 54.6604233, -133.6341553, 133.5435181
17: -110.5609436, 47.5792503, -110.6021347, 47.6870346, -158.2479706, 158.1813812
18: -78.7295837, 54.1633835, -78.7789154, 54.2670784, -132.9966583, 132.9422913
19: -57.5772057, 35.9203415, -57.6239624, 36.0351486, -93.6123352, 93.5443039
20: -56.2872849, 39.5818024, -56.3359413, 39.7074127, -95.9946823, 95.9177399
21: -73.8327789, 41.3040771, -73.8793030, 41.4628830, -115.2956619, 115.1833801
22: -68.8582230, 43.9032135, -68.8994293, 43.9809761, -112.8391953, 112.8026428
23: -61.4061775, 46.5395050, -61.4437027, 46.6379242, -108.0440979, 107.9832001
24: -73.2307129, 46.1248322, -73.2848587, 46.1619873, -119.3926849, 119.4096909
25: -64.0237274, 47.3331070, -64.0593414, 47.4352722, -111.4589767, 111.3924484
26: -82.7070160, 61.4796524, -82.7593384, 61.6923752, -144.3993835, 144.2389832
27: -69.0891266, 45.8833313, -69.1685257, 45.9175262, -115.0066528, 115.0518570
28: -58.2086182, 48.6852188, -58.2492447, 48.7820435, -106.9906616, 106.9344635
29: -74.9307251, 42.0965996, -74.9634247, 42.1826973, -117.1134186, 117.0600052
30: -78.8560181, 47.6862679, -78.8979340, 47.8069153, -126.6629181, 126.5841980
31: -79.9453430, 47.6296501, -80.0063477, 47.7676735, -127.7130127, 127.6360016
32: -83.3446808, 42.5247154, -83.3878479, 42.6250763, -125.9697571, 125.9125671
33: -109.5497742, 51.9672089, -109.6715927, 52.0160866, -161.5658569, 161.6387939
34: -97.5978851, 28.3786812, -97.6780701, 28.4234734, -126.0213623, 126.0567398
35: -91.3086853, 39.5517616, -91.3882751, 39.5897446, -130.8984222, 130.9400330
36: -89.8961334, 45.4528236, -89.9438934, 45.4960632, -135.3921967, 135.3967133
37: -131.2169189, 40.2779846, -131.2815857, 40.3558578, -171.5727692, 171.5595703
38: -106.5425110, 49.4991035, -106.6358871, 49.5420074, -156.0845184, 156.1349792
39: -118.3792419, 57.0401192, -118.4603119, 57.1100845, -175.4893188, 175.5004272
40: -99.9716492, 35.2111969, -100.0369568, 35.2432899, -135.2149353, 135.2481537
41: -84.0579147, 51.0648232, -84.1132507, 51.1132736, -135.1711884, 135.1780701
42: -66.1099396, 37.9253464, -66.1534119, 38.0112839, -104.1212082, 104.0787582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1415
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
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1552
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
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5746548, upper bound: 74.6022737
time: 82.88 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5746548, upper bound: 74.6416266
time: 110.76 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -91.4338989, 65.8267288, -91.3442154, 65.7612381, -157.1951294, 157.1709442
1: -45.6207848, 56.1456070, -45.5757980, 56.0146675, -101.6354446, 101.7214050
2: -39.8771133, 57.2357330, -39.8625755, 57.0522690, -96.9293823, 97.0983047
3: -49.9112701, 59.2532692, -49.9391632, 59.0894203, -109.0006866, 109.1924286
4: -48.7564049, 73.1786346, -48.7419014, 72.9658813, -121.7222748, 121.9205246
5: -46.1513596, 58.1457367, -46.1711044, 58.0077209, -104.1590729, 104.3168411
6: -91.0247345, 43.7854691, -90.8203888, 43.7371330, -134.7618408, 134.6058502
7: -54.8932571, 56.7288094, -54.8259506, 56.6470795, -111.5403290, 111.5547638
8: -60.6793213, 82.8389130, -60.6285019, 82.5982971, -143.2776031, 143.4674072
9: -49.6666260, 63.5584526, -49.4241219, 63.4728088, -113.1394348, 112.9825745
10: -77.1121521, 71.9176483, -76.5018768, 71.8117523, -148.9239044, 148.4195251
11: -81.1769180, 37.4926949, -80.5612640, 37.4332199, -118.6101379, 118.0539551
12: -85.1918182, 51.1994934, -84.7113342, 51.1151428, -136.3069611, 135.9108276
13: -77.5595093, 80.7984161, -77.5047150, 80.6067505, -158.1662598, 158.3031311
14: -117.7800751, 55.6163406, -117.2165604, 55.5062904, -173.2863617, 172.8329010
15: -60.5569153, 63.3658943, -60.4099274, 63.1210670, -123.6779785, 123.7758179
16: -79.4862671, 54.7474098, -79.0490799, 54.6604233, -134.1466980, 133.7964935
17: -110.8771286, 47.8085785, -110.6021347, 47.6870346, -158.5641327, 158.4107056
18: -79.0050049, 54.3086586, -78.7789154, 54.2670784, -133.2720795, 133.0875702
19: -57.7675934, 36.0266876, -57.6239624, 36.0351486, -93.8027420, 93.6506500
20: -56.5647888, 39.7293396, -56.3359413, 39.7074127, -96.2722015, 96.0652771
21: -74.2364197, 41.4875984, -73.8793030, 41.4628830, -115.6993027, 115.3668976
22: -69.0749054, 44.1135368, -68.8994293, 43.9809761, -113.0558777, 113.0129700
23: -61.5722313, 46.6534843, -61.4437027, 46.6379242, -108.2101593, 108.0971756
24: -73.4411926, 46.1718292, -73.2848587, 46.1619873, -119.6031647, 119.4566879
25: -64.1577988, 47.4803467, -64.0593414, 47.4352722, -111.5930710, 111.5396881
26: -82.9858704, 61.7168350, -82.7593384, 61.6923752, -144.6782532, 144.4761658
27: -69.3677979, 45.9666443, -69.1685257, 45.9175262, -115.2853088, 115.1351700
28: -58.3462563, 48.7567825, -58.2492447, 48.7820435, -107.1282959, 107.0060272
29: -75.1340790, 42.2264862, -74.9634247, 42.1826973, -117.3167725, 117.1898956
30: -79.0244064, 47.8249283, -78.8979340, 47.8069153, -126.8313141, 126.7228622
31: -80.2249451, 47.7910309, -80.0063477, 47.7676735, -127.9926147, 127.7973785
32: -83.6553040, 42.6757812, -83.3878479, 42.6250763, -126.2803802, 126.0636292
33: -109.8073120, 52.2885361, -109.6715927, 52.0160866, -161.8233643, 161.9601288
34: -97.7927094, 28.6629028, -97.6780701, 28.4234734, -126.2161865, 126.3409653
35: -91.5038528, 39.8465118, -91.3882751, 39.5897446, -131.0935974, 131.2347870
36: -90.0530090, 45.5732002, -89.9438934, 45.4960632, -135.5490723, 135.5170898
37: -131.4750671, 40.4192009, -131.2815857, 40.3558578, -171.8308868, 171.7007751
38: -106.7706680, 49.6926270, -106.6358871, 49.5420074, -156.3126831, 156.3284912
39: -118.6032639, 57.2369156, -118.4603119, 57.1100845, -175.7133484, 175.6972351
40: -100.2222214, 35.3467407, -100.0369568, 35.2432899, -135.4655151, 135.3836975
41: -84.2253418, 51.1602936, -84.1132507, 51.1132736, -135.3386230, 135.2735443
42: -66.3718262, 38.0065536, -66.1534119, 38.0112839, -104.3831100, 104.1599579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
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
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1415
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
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 807
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
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1552
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
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5746548, upper bound: 74.6022737
time: 111.84 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5746548, upper bound: 74.6416266
time: 147.83 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -91.2255783, 65.7379913, -91.5687408, 65.8625946, -157.0881653, 157.3067322
1: -45.4698753, 55.9971924, -45.7169609, 56.1666794, -101.6365509, 101.7141495
2: -39.7195740, 57.0323830, -40.0259094, 57.2561989, -96.9757538, 97.0582886
3: -49.7656059, 59.0669060, -50.0751953, 59.2861023, -109.0517120, 109.1421051
4: -48.5776138, 72.9528656, -48.9229889, 73.2050781, -121.7826843, 121.8758545
5: -45.9996872, 57.9934578, -46.3294525, 58.1773911, -104.1770782, 104.3229065
6: -90.7939911, 43.7122917, -91.0797882, 43.8433151, -134.6372986, 134.7920837
7: -54.6851768, 56.6420898, -55.0224533, 56.7519455, -111.4371185, 111.6645432
8: -60.5163918, 82.5781097, -60.8351669, 82.8705521, -143.3869324, 143.4132690
9: -49.3788986, 63.4613342, -49.7112427, 63.6796722, -113.0585632, 113.1725769
10: -76.4557419, 71.6852875, -77.1761017, 72.1721420, -148.6278839, 148.8613892
11: -80.5377960, 37.2910385, -81.2290421, 37.6734848, -118.2112808, 118.5200806
12: -84.6851807, 50.9785042, -85.2292328, 51.4562225, -136.1414032, 136.2077332
13: -77.3660889, 80.6179962, -77.6259766, 80.8753281, -158.2414246, 158.2439575
14: -117.1512909, 55.3699837, -117.8489609, 55.8435555, -172.9948425, 173.2189484
15: -60.3501282, 63.0925941, -60.6819038, 63.4129333, -123.7630615, 123.7744980
16: -78.9817200, 54.5889969, -79.5600433, 54.9125633, -133.8942871, 134.1490479
17: -110.5587692, 47.5823135, -110.9180679, 47.9189186, -158.4776917, 158.5003815
18: -78.7424316, 54.1271667, -79.0548401, 54.4122772, -133.1547089, 133.1820068
19: -57.5895195, 35.8845444, -57.8144417, 36.1409569, -93.7304764, 93.6989822
20: -56.3031578, 39.6001434, -56.6130638, 39.8545227, -96.1576843, 96.2132034
21: -73.8469238, 41.2920456, -74.2826080, 41.6457558, -115.4926605, 115.5746536
22: -68.9327393, 43.8965111, -69.1165771, 44.1905785, -113.1233215, 113.0130920
23: -61.4146271, 46.4935608, -61.6098404, 46.7514496, -108.1660767, 108.1034012
24: -73.3191833, 46.0590935, -73.4954681, 46.2090759, -119.5282440, 119.5545578
25: -64.0456238, 47.3077431, -64.1938095, 47.5824814, -111.6280975, 111.5015411
26: -82.7284241, 61.4683685, -83.0382156, 61.9307175, -144.6591492, 144.5065918
27: -69.2093353, 45.8429756, -69.4470749, 46.0006256, -115.2099609, 115.2900543
28: -58.2474327, 48.6314621, -58.3871651, 48.8535843, -107.1010132, 107.0186234
29: -74.9808502, 42.0693741, -75.1670837, 42.3110809, -117.2919312, 117.2364578
30: -78.8780212, 47.6336441, -79.0665359, 47.9454193, -126.8234406, 126.7001724
31: -79.9623642, 47.6092262, -80.2853851, 47.9286728, -127.8910294, 127.8946075
32: -83.3655701, 42.6055641, -83.6980743, 42.7757263, -126.1412811, 126.3036346
33: -109.6233673, 51.9855003, -109.9289627, 52.3369980, -161.9603577, 161.9144592
34: -97.6520920, 28.3833275, -97.8729172, 28.7072792, -126.3593750, 126.2562408
35: -91.3762817, 39.5692368, -91.5833435, 39.8834801, -131.2597656, 131.1525726
36: -89.9025269, 45.4608955, -90.1008530, 45.6177292, -135.5202332, 135.5617371
37: -131.2956543, 40.2678452, -131.5399475, 40.4960938, -171.7917480, 171.8078003
38: -106.5398331, 49.5534668, -106.8631744, 49.7362900, -156.2761230, 156.4166412
39: -118.3887939, 57.0965233, -118.6855927, 57.3067245, -175.6955261, 175.7821198
40: -100.0104446, 35.2284966, -100.2871246, 35.3803291, -135.3907776, 135.5156250
41: -84.0685272, 51.0642624, -84.2806015, 51.2076874, -135.2762146, 135.3448486
42: -66.1401825, 37.9449081, -66.4153137, 38.0940666, -104.2342529, 104.3602142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1544
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
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1565
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
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1382
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4919848, upper bound: 74.5830550
time: 99.91 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4919848, upper bound: 74.6331884
time: 91.94 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -91.4307861, 65.8417664, -91.5473328, 65.8528595, -157.2836456, 157.3890991
1: -45.6090279, 56.3023758, -45.7036934, 56.1623001, -101.7713165, 102.0060730
2: -39.8670082, 57.3613777, -40.0124435, 57.2506485, -97.1176605, 97.3738174
3: -49.9127045, 59.5311928, -50.0588837, 59.2782440, -109.1909485, 109.5900726
4: -48.7702980, 73.3890228, -48.9070892, 73.1993332, -121.9696350, 122.2961121
5: -46.1619873, 58.3446541, -46.3147469, 58.1705933, -104.3325806, 104.6594009
6: -90.9687042, 43.7618752, -91.0690918, 43.8078995, -134.7765961, 134.8309631
7: -54.8779106, 56.8361320, -55.0092773, 56.7468262, -111.6247406, 111.8454056
8: -60.6699066, 82.9754944, -60.8219757, 82.8622513, -143.5321655, 143.7974548
9: -49.4979858, 63.6100388, -49.6984406, 63.6688499, -113.1668243, 113.3084717
10: -77.0754318, 71.9561005, -77.1671448, 72.1472778, -149.2227173, 149.1232147
11: -81.4521561, 37.4709854, -81.2161865, 37.6511612, -119.1033096, 118.6871719
12: -85.1030731, 51.1841354, -85.2218475, 51.4365311, -136.5396118, 136.4059753
13: -77.4699402, 81.0709686, -77.5850830, 80.8606110, -158.3305511, 158.6560516
14: -117.9264984, 55.5725098, -117.8344421, 55.8177376, -173.7442322, 173.4069519
15: -60.4730186, 63.4248619, -60.6381226, 63.4040756, -123.8770752, 124.0629730
16: -79.5068512, 54.7375946, -79.5428162, 54.8731995, -134.3800507, 134.2804108
17: -111.2715454, 47.7601700, -110.9083557, 47.8985634, -159.1701050, 158.6685181
18: -79.3429184, 54.2741013, -79.0421677, 54.3827171, -133.7256317, 133.3162689
19: -58.1467247, 36.0161209, -57.8065948, 36.1266479, -94.2733765, 93.8227158
20: -56.6804848, 39.7121887, -56.6056252, 39.8385010, -96.5189819, 96.3178024
21: -74.5813446, 41.4611778, -74.2741013, 41.6278610, -116.2092056, 115.7352753
22: -69.2369232, 44.0120277, -69.1094208, 44.1752167, -113.4121399, 113.1214447
23: -61.9064636, 46.6406555, -61.6029434, 46.7353745, -108.6418381, 108.2435837
24: -73.6568375, 46.1270065, -73.4829025, 46.1828156, -119.8396530, 119.6099091
25: -64.4238434, 47.4328537, -64.1837234, 47.5667229, -111.9905701, 111.6165771
26: -83.2683105, 61.6988029, -83.0270081, 61.9077225, -145.1760254, 144.7257996
27: -69.4864044, 45.9141693, -69.4353790, 45.9851456, -115.4715500, 115.3495407
28: -58.5633507, 48.7511978, -58.3805199, 48.8395233, -107.4028778, 107.1317139
29: -75.4389038, 42.1774902, -75.1596985, 42.2986183, -117.7375183, 117.3371811
30: -79.4325562, 47.8033180, -79.0544434, 47.9260216, -127.3585815, 126.8577576
31: -80.6524811, 47.7574539, -80.2750244, 47.9091988, -128.5616608, 128.0324707
32: -83.4940643, 42.7003326, -83.6805420, 42.7581787, -126.2522430, 126.3808594
33: -109.7987823, 52.3930168, -109.9101868, 52.3296928, -162.1284790, 162.3032074
34: -97.7826843, 28.5935783, -97.8597870, 28.6978416, -126.4805298, 126.4533539
35: -91.4643250, 39.8466682, -91.5629730, 39.8785706, -131.3428955, 131.4096375
36: -90.0147018, 45.6597214, -90.0812302, 45.6124802, -135.6271820, 135.7409363
37: -131.4942169, 40.3762589, -131.5197449, 40.4706612, -171.9648743, 171.8959808
38: -106.7398376, 49.8313713, -106.8453445, 49.7286987, -156.4685364, 156.6767120
39: -118.5798340, 57.3289032, -118.6630707, 57.3003883, -175.8802185, 175.9919739
40: -100.2068481, 35.4186554, -100.2731018, 35.3749237, -135.5817719, 135.6917572
41: -84.2109680, 51.1937714, -84.2637711, 51.1936646, -135.4046173, 135.4575500
42: -66.2932968, 38.0618515, -66.4089966, 38.0625038, -104.3558044, 104.4708405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 917
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5198832, upper bound: 74.5533824
time: 140.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5198832, upper bound: 74.6031303
time: 652.23 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -91.4449539, 65.9252396, -91.3644485, 65.7664185, -157.2113647, 157.2896881
1: -45.6409302, 56.2520599, -45.5914078, 56.0175934, -101.6585236, 101.8434677
2: -39.9365005, 57.3253899, -39.8885117, 57.0544319, -96.9909363, 97.2138977
3: -50.0022011, 59.3907242, -49.9652939, 59.0934334, -109.0956345, 109.3560104
4: -48.8356628, 73.2730789, -48.7711105, 72.9682159, -121.8038788, 122.0441895
5: -46.2602158, 58.2741623, -46.2035370, 58.0116959, -104.2719116, 104.4776917
6: -90.9544067, 43.7695770, -90.8274078, 43.7288589, -134.6832581, 134.5969849
7: -54.9345627, 56.8392105, -54.8464661, 56.6496048, -111.5841675, 111.6856766
8: -60.7100029, 82.9263229, -60.6552048, 82.6026154, -143.3125916, 143.5815277
9: -49.6254501, 63.5656586, -49.4295883, 63.4939766, -113.1194305, 112.9952469
10: -76.9538422, 71.9273148, -76.5107880, 71.8557281, -148.8095703, 148.4381104
11: -81.0139236, 37.5021553, -80.5689316, 37.4632912, -118.4771957, 118.0710831
12: -85.1374435, 51.2231941, -84.7160034, 51.1607056, -136.2981567, 135.9391937
13: -77.5410004, 80.7319260, -77.5024109, 80.6179199, -158.1589203, 158.2343140
14: -117.6174774, 55.5926895, -117.2262115, 55.5458832, -173.1633606, 172.8188934
15: -60.4865265, 63.2941055, -60.4118919, 63.1283455, -123.6148682, 123.7059860
16: -79.3474274, 54.7443123, -79.0583191, 54.6814766, -134.0289001, 133.8026276
17: -110.8593674, 47.7776108, -110.6077652, 47.7022552, -158.5616150, 158.3853760
18: -79.0143890, 54.3303986, -78.7862778, 54.2829971, -133.2973785, 133.1166687
19: -57.8714981, 36.0820007, -57.6309433, 36.0536118, -93.9251099, 93.7129440
20: -56.5466843, 39.7505951, -56.3434258, 39.7275276, -96.2742157, 96.0940247
21: -74.2442932, 41.5343704, -73.8853455, 41.4900475, -115.7343445, 115.4197083
22: -69.0275269, 44.0414352, -68.9044952, 43.9841843, -113.0117111, 112.9459305
23: -61.6306419, 46.6955452, -61.4488258, 46.6545258, -108.2851639, 108.1443634
24: -73.3572845, 46.1751747, -73.2879868, 46.1596832, -119.5169678, 119.4631653
25: -64.1822510, 47.5038452, -64.0632858, 47.4520111, -111.6342545, 111.5671310
26: -83.0511169, 61.7912903, -82.7657013, 61.7291107, -144.7802277, 144.5569763
27: -69.2792740, 45.9372597, -69.1801987, 45.9178696, -115.1971436, 115.1174622
28: -58.3719025, 48.8310738, -58.2553368, 48.7980919, -107.1699982, 107.0864029
29: -75.1363373, 42.2285957, -74.9664917, 42.1904984, -117.3268280, 117.1950836
30: -79.1091385, 47.8803902, -78.9028320, 47.8267555, -126.9358978, 126.7832184
31: -80.2865448, 47.8203468, -80.0160675, 47.7892914, -128.0758362, 127.8364105
32: -83.5951691, 42.6823273, -83.3932266, 42.6410751, -126.2362442, 126.0755463
33: -109.7476959, 52.2033806, -109.6905594, 52.0235901, -161.7712708, 161.8939362
34: -97.7438889, 28.5384750, -97.6900482, 28.4299545, -126.1738434, 126.2285233
35: -91.4281387, 39.7405624, -91.3974457, 39.5963097, -131.0244446, 131.1380005
36: -90.0135880, 45.5616112, -89.9476547, 45.4985085, -135.5121002, 135.5092621
37: -131.3998413, 40.4261589, -131.2871857, 40.3640671, -171.7639160, 171.7133484
38: -106.7448425, 49.6454964, -106.6468735, 49.5467796, -156.2915955, 156.2923584
39: -118.5673676, 57.2114906, -118.4695435, 57.1190186, -175.6863861, 175.6810303
40: -100.1728668, 35.3180695, -100.0458755, 35.2423248, -135.4151764, 135.3639374
41: -84.2074280, 51.1603661, -84.1203766, 51.1139908, -135.3214111, 135.2807465
42: -66.2867126, 38.0489616, -66.1585236, 38.0012512, -104.2879639, 104.2074890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 924
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
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1467
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
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5746548, upper bound: 74.5687592
time: 93.60 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5746548, upper bound: 74.5687592
time: 85.08 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -91.6718216, 66.0245819, -91.3644485, 65.7664185, -157.4382324, 157.3890228
1: -45.7816429, 56.4059563, -45.5914078, 56.0175934, -101.7992401, 101.9973602
2: -40.1045761, 57.5324249, -39.8885117, 57.0544319, -97.1590042, 97.4209290
3: -50.1450996, 59.5936012, -49.9652939, 59.0934334, -109.2385254, 109.5588913
4: -49.0149269, 73.5174255, -48.7711105, 72.9682159, -121.9831390, 122.2885284
5: -46.4194183, 58.4483604, -46.2035370, 58.0116959, -104.4311142, 104.6518860
6: -91.2180634, 43.8772163, -90.8274078, 43.7288589, -134.9469299, 134.7046204
7: -55.1306305, 56.9439621, -54.8464661, 56.6496048, -111.7802353, 111.7904205
8: -60.9159775, 83.2005844, -60.6552048, 82.6026154, -143.5185852, 143.8557892
9: -49.9140930, 63.7708130, -49.4295883, 63.4939766, -113.4080658, 113.2004013
10: -77.6290283, 72.2845001, -76.5107880, 71.8557281, -149.4847412, 148.7952881
11: -81.6827850, 37.7422523, -80.5689316, 37.4632912, -119.1460724, 118.3111801
12: -85.6632385, 51.5669708, -84.7160034, 51.1607056, -136.8239441, 136.2829590
13: -77.6618881, 80.9976273, -77.5024109, 80.6179199, -158.2798157, 158.5000305
14: -118.2550354, 55.9350395, -117.2262115, 55.5458832, -173.8009033, 173.1612549
15: -60.7568092, 63.6019592, -60.4118919, 63.1283455, -123.8851547, 124.0138245
16: -79.8634109, 54.9945297, -79.0583191, 54.6814766, -134.5448914, 134.0528564
17: -111.1762238, 48.0081329, -110.6077652, 47.7022552, -158.8784485, 158.6158905
18: -79.2881393, 54.4795609, -78.7862778, 54.2829971, -133.5711212, 133.2658386
19: -58.0619698, 36.1861992, -57.6309433, 36.0536118, -94.1155853, 93.8171387
20: -56.8260345, 39.8969650, -56.3434258, 39.7275276, -96.5535583, 96.2403870
21: -74.6495361, 41.7154312, -73.8853455, 41.4900475, -116.1395721, 115.6007767
22: -69.2457123, 44.2521896, -68.9044952, 43.9841843, -113.2298889, 113.1566772
23: -61.7992744, 46.8079529, -61.4488258, 46.6545258, -108.4537964, 108.2567749
24: -73.5679626, 46.2250595, -73.2879868, 46.1596832, -119.7276459, 119.5130386
25: -64.3129425, 47.6502686, -64.0632858, 47.4520111, -111.7649536, 111.7135544
26: -83.3349991, 62.0406837, -82.7657013, 61.7291107, -145.0641174, 144.8063660
27: -69.5550232, 46.0206909, -69.1801987, 45.9178696, -115.4728928, 115.2008896
28: -58.5050430, 48.9024734, -58.2553368, 48.7980919, -107.3031311, 107.1577988
29: -75.3362579, 42.3569527, -74.9664917, 42.1904984, -117.5267487, 117.3234406
30: -79.2793808, 48.0185242, -78.9028320, 47.8267555, -127.1061401, 126.9213562
31: -80.5700684, 47.9806747, -80.0160675, 47.7892914, -128.3593597, 127.9967346
32: -83.9075317, 42.8308525, -83.3932266, 42.6410751, -126.5485916, 126.2240753
33: -110.0066528, 52.5219421, -109.6905594, 52.0235901, -162.0302429, 162.2124939
34: -97.9383545, 28.8202896, -97.6900482, 28.4299545, -126.3683014, 126.5103378
35: -91.6235504, 40.0364037, -91.3974457, 39.5963097, -131.2198639, 131.4338379
36: -90.1709671, 45.6842499, -89.9476547, 45.4985085, -135.6694641, 135.6318970
37: -131.6589966, 40.5643921, -131.2871857, 40.3640671, -172.0230713, 171.8515778
38: -106.9674683, 49.8391037, -106.6468735, 49.5467796, -156.5142517, 156.4859619
39: -118.7967148, 57.4076843, -118.4695435, 57.1190186, -175.9157257, 175.8772278
40: -100.4262314, 35.4531136, -100.0458755, 35.2423248, -135.6685486, 135.4989777
41: -84.3760071, 51.2514572, -84.1203766, 51.1139908, -135.4899902, 135.3718262
42: -66.5541000, 38.1414108, -66.1585236, 38.0012512, -104.5553513, 104.2999344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1623
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
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1555
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
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 924
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
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
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
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
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
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1467
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
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5896907, upper bound: 74.5687592
time: 93.47 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5896907, upper bound: 74.5687592
time: 96.12 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -91.4595184, 65.9358826, -91.5887833, 65.8679276, -157.3274384, 157.5246582
1: -45.6315651, 56.2559204, -45.7321587, 56.1696243, -101.8011932, 101.9880676
2: -39.9446411, 57.3259506, -40.0514450, 57.2583466, -97.2029877, 97.3773956
3: -49.9955673, 59.4010582, -50.1055908, 59.2901154, -109.2856750, 109.5066376
4: -48.8381004, 73.2865219, -48.9518051, 73.2076874, -122.0457916, 122.2383270
5: -46.2681732, 58.2912521, -46.3634453, 58.1812553, -104.4494324, 104.6546936
6: -90.9832153, 43.8011665, -91.0870209, 43.8350487, -134.8182678, 134.8881836
7: -54.9248505, 56.8561325, -55.0419273, 56.7545624, -111.6794128, 111.8980560
8: -60.7540665, 82.9382019, -60.8615723, 82.8747253, -143.6287689, 143.7997742
9: -49.6251526, 63.6750031, -49.7168159, 63.7004929, -113.3256378, 113.3918152
10: -76.9719849, 72.0555420, -77.1849060, 72.2151794, -149.1871643, 149.2404327
11: -81.0426178, 37.5408936, -81.2365417, 37.7034760, -118.7460785, 118.7774353
12: -85.1488800, 51.3429565, -85.2340927, 51.5014343, -136.6503143, 136.5770416
13: -77.4676056, 80.8163757, -77.6231232, 80.8868713, -158.3544769, 158.4394989
14: -117.6197968, 55.6828842, -117.8586426, 55.8830185, -173.5028076, 173.5415344
15: -60.5428963, 63.3136787, -60.6839142, 63.4199524, -123.9628448, 123.9975891
16: -79.3547134, 54.8336868, -79.5691757, 54.9332314, -134.2879486, 134.4028625
17: -110.8561401, 47.7823715, -110.9235535, 47.9346008, -158.7907410, 158.7059021
18: -79.0251694, 54.2950821, -79.0621643, 54.4291534, -133.4543152, 133.3572388
19: -57.8826866, 36.0453491, -57.8213577, 36.1591263, -94.0418091, 93.8666992
20: -56.5623856, 39.7683334, -56.6205521, 39.8743515, -96.4367371, 96.3888855
21: -74.2580261, 41.5215416, -74.2886353, 41.6725464, -115.9305725, 115.8101730
22: -69.1018982, 44.0348587, -69.1218414, 44.1939316, -113.2958298, 113.1567001
23: -61.6385117, 46.6492004, -61.6150169, 46.7677155, -108.4062119, 108.2642212
24: -73.4442291, 46.1069717, -73.4987106, 46.2079086, -119.6521378, 119.6056671
25: -64.1999817, 47.4783859, -64.1977386, 47.5991402, -111.7991180, 111.6761246
26: -83.0715485, 61.7872238, -83.0448456, 61.9720612, -145.0436096, 144.8320618
27: -69.3979874, 45.8967056, -69.4585571, 46.0009308, -115.3989182, 115.3552628
28: -58.4102554, 48.7779579, -58.3932152, 48.8695107, -107.2797699, 107.1711578
29: -75.1815338, 42.1996918, -75.1703491, 42.3188019, -117.5003052, 117.3700256
30: -79.1302567, 47.8272629, -79.0713959, 47.9652786, -127.0955353, 126.8986588
31: -80.3032532, 47.7994995, -80.2949829, 47.9502945, -128.2535400, 128.0944824
32: -83.6162262, 42.7610207, -83.7035294, 42.7914314, -126.4076538, 126.4645386
33: -109.8209610, 52.2202682, -109.9478836, 52.3443680, -162.1653137, 162.1681519
34: -97.7977371, 28.5423927, -97.8847656, 28.7134552, -126.5111923, 126.4271545
35: -91.4945068, 39.7570763, -91.5923767, 39.8897209, -131.3842316, 131.3494568
36: -90.0199432, 45.5706482, -90.1048126, 45.6210785, -135.6410217, 135.6754608
37: -131.4776611, 40.4141273, -131.5457611, 40.5037193, -171.9813843, 171.9598846
38: -106.7417908, 49.6991768, -106.8725281, 49.7414856, -156.4832764, 156.5717010
39: -118.5802078, 57.2692604, -118.6960297, 57.3167953, -175.8970032, 175.9652863
40: -100.2111664, 35.3351707, -100.2960663, 35.3797379, -135.5909119, 135.6312408
41: -84.2188721, 51.1582680, -84.2877808, 51.2070999, -135.4259644, 135.4460449
42: -66.3166962, 38.0699005, -66.4207611, 38.0900002, -104.4066925, 104.4906464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1544
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
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1565
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
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 577
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
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 736
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.5552377, upper bound: 74.5366978
time: 95.05 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4919848, upper bound: 74.5366978
time: 93.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 191.07 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5746548, upper bound: 74.6022737
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5746548, upper bound: 74.6416266
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5746548, upper bound: 74.6022737
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5746548, upper bound: 74.6416266
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.4919848, upper bound: 74.5830550
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.4919848, upper bound: 74.6331884
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5198832, upper bound: 74.5533824
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5198832, upper bound: 74.6031303
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5746548, upper bound: 74.5687592
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5746548, upper bound: 74.5687592
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5896907, upper bound: 74.5687592
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5896907, upper bound: 74.5687592
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.5552377, upper bound: 74.5366978
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 191.07
Output dim: 4, lower bound: -74.4919848, upper bound: 74.5366978
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 191.07
Output dim: 4, lower bound: -74.6031299, upper bound: 74.6031303
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=121.93731689453125
rel_dist={4: [-74.7091100354215, 74.70911000353107]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 12456.10 seconds
