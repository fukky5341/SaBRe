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
execution time: IAR + LP analysis = 2.77 + 102.78 = 105.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -80.9148816, upper bound: 80.9148816


# Binary Search by BASE starts (time budget: 17894.44 seconds, max iter: 100)

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
Binary search time: 839.15 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 17055.29 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.9300824, upper bound: 77.9300824
time: 96.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.9300824, upper bound: 77.9300824
time: 94.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 190.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 190.86
Output dim: 4, lower bound: -77.9300824, upper bound: 77.9300824
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 190.86
Output dim: 4, lower bound: -77.9300824, upper bound: 77.9300824

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.05 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1673

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.8126146, upper bound: 77.8126146
time: 77.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.8126146, upper bound: 77.8126146
time: 385.71 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.8126146, upper bound: 77.8126146
time: 839.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.8126146, upper bound: 77.8126146
time: 370.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1211.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1211.56
Output dim: 4, lower bound: -77.8126146, upper bound: 77.8126146
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1211.56
Output dim: 4, lower bound: -77.8126146, upper bound: 77.8126146
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1211.56
Output dim: 4, lower bound: -77.8126146, upper bound: 77.8126146
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1211.56
Output dim: 4, lower bound: -77.8126146, upper bound: 77.8126146

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
time: 97.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
time: 246.42 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
time: 97.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
time: 101.26 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.06 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
time: 97.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
time: 254.59 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

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
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
time: 99.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
time: 103.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 205.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.39
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.39
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.39
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.39
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.39
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.39
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 205.39
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 205.39
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.6267759, upper bound: 77.8055853
time: 84.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -77.7140212, upper bound: 77.7191241
time: 928.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1014.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1014.93
Output dim: 4, lower bound: -77.6267759, upper bound: 77.8055853
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1014.93
Output dim: 4, lower bound: -77.7140212, upper bound: 77.7191241
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1014.93
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1014.93
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1014.93
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1014.93
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1014.93
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1014.93
Output dim: 4, lower bound: -77.7175310, upper bound: 77.8089450
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1014.93
Output dim: 4, lower bound: -77.8089450, upper bound: 77.7175310
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=121.93731689453125
rel_dist={4: [-78.19466474792833, 78.1946647699462]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7176366, upper bound: 75.7176366
time: 218.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.7176366, upper bound: 75.7176366
time: 135.44 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 354.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 354.55
Output dim: 4, lower bound: -75.7176366, upper bound: 75.7176366
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 354.55
Output dim: 4, lower bound: -75.7176366, upper bound: 75.7176366

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.6302394, upper bound: 75.6302394
time: 921.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.6302394, upper bound: 75.6302394
time: 937.48 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
Output dim: 4, lower bound: -75.6302394, upper bound: 75.6302394
time: 929.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -75.6302394, upper bound: 75.6302394
time: 946.46 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1878.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1878.28
Output dim: 4, lower bound: -75.6302394, upper bound: 75.6302394
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1878.28
Output dim: 4, lower bound: -75.6302394, upper bound: 75.6302394
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1878.28
Output dim: 4, lower bound: -75.6302394, upper bound: 75.6302394
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1878.28
Output dim: 4, lower bound: -75.6302394, upper bound: 75.6302394
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=121.93731689453125
rel_dist={4: [-75.94959621478742, 75.94959622088305]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4880522, upper bound: 74.4880521
time: 86.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4880522, upper bound: 74.4880521
time: 87.15 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 174.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 174.22
Output dim: 4, lower bound: -74.4880522, upper bound: 74.4880521
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 174.22
Output dim: 4, lower bound: -74.4880522, upper bound: 74.4880521

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4048229, upper bound: 74.4048228
time: 97.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4048229, upper bound: 74.4048228
time: 98.02 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

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
Output dim: 4, lower bound: -74.4048229, upper bound: 74.4048228
time: 95.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4048229, upper bound: 74.4048228
time: 96.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 194.87 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 194.87
Output dim: 4, lower bound: -74.4048229, upper bound: 74.4048228
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 194.87
Output dim: 4, lower bound: -74.4048229, upper bound: 74.4048228
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 194.87
Output dim: 4, lower bound: -74.4048229, upper bound: 74.4048228
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 194.87
Output dim: 4, lower bound: -74.4048229, upper bound: 74.4048228

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.3319820, upper bound: 74.4020338
time: 84.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4020338, upper bound: 74.3319820
time: 112.17 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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
Output dim: 4, lower bound: -74.3319820, upper bound: 74.4020338
time: 81.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4020338, upper bound: 74.3319820
time: 113.50 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.3319820, upper bound: 74.4020338
time: 84.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4020338, upper bound: 74.3319820
time: 126.32 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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
Output dim: 4, lower bound: -74.3319820, upper bound: 74.4020338
time: 82.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -74.4020338, upper bound: 74.3319820
time: 126.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 211.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 211.36
Output dim: 4, lower bound: -74.3319820, upper bound: 74.4020338
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 211.36
Output dim: 4, lower bound: -74.4020338, upper bound: 74.3319820
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 211.36
Output dim: 4, lower bound: -74.3319820, upper bound: 74.4020338
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 211.36
Output dim: 4, lower bound: -74.4020338, upper bound: 74.3319820
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 211.36
Output dim: 4, lower bound: -74.3319820, upper bound: 74.4020338
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 211.36
Output dim: 4, lower bound: -74.4020338, upper bound: 74.3319820
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 211.36
Output dim: 4, lower bound: -74.3319820, upper bound: 74.4020338
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 211.36
Output dim: 4, lower bound: -74.4020338, upper bound: 74.3319820

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.2651926, upper bound: 74.3994857
time: 107.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3295221, upper bound: 74.3358937
time: 91.00 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3358937, upper bound: 74.3295221
time: 89.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3994857, upper bound: 74.2651926
time: 95.09 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.2651926, upper bound: 74.3994857
time: 107.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3295221, upper bound: 74.3358937
time: 84.20 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3358937, upper bound: 74.3295221
time: 87.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3994857, upper bound: 74.2651926
time: 97.07 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.2651926, upper bound: 74.3994857
time: 110.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3295221, upper bound: 74.3358937
time: 94.59 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

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
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3358937, upper bound: 74.3295221
time: 90.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3994857, upper bound: 74.2651926
time: 93.56 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

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
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.2651926, upper bound: 74.3994857
time: 106.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3295221, upper bound: 74.3358937
time: 91.86 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3358937, upper bound: 74.3295221
time: 90.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -74.3994857, upper bound: 74.2651926
time: 96.84 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 189.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.2651926, upper bound: 74.3994857
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3295221, upper bound: 74.3358937
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3358937, upper bound: 74.3295221
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3994857, upper bound: 74.2651926
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.2651926, upper bound: 74.3994857
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3295221, upper bound: 74.3358937
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3358937, upper bound: 74.3295221
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3994857, upper bound: 74.2651926
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.2651926, upper bound: 74.3994857
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3295221, upper bound: 74.3358937
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3358937, upper bound: 74.3295221
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3994857, upper bound: 74.2651926
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.2651926, upper bound: 74.3994857
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3295221, upper bound: 74.3358937
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3358937, upper bound: 74.3295221
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 189.80
Output dim: 4, lower bound: -74.3994857, upper bound: 74.2651926
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=121.93731689453125
rel_dist={4: [-74.7091100354215, 74.70911000353107]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 12150.77 seconds
