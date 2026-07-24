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
execution time: IAR + RelationalAnalysis = 2.97 + 93.60 = 96.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -70.3169252, upper bound: 70.3169252

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 819

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 642

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2809260, upper bound: 70.3090181
time: 74.18 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3090181, upper bound: 70.2809260
time: 115.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 190.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 190.11
Output dim: 4, lower bound: -70.2809260, upper bound: 70.3090181
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 190.11
Output dim: 4, lower bound: -70.3090181, upper bound: 70.2809260

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

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 898

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 861

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2690263, upper bound: 70.3081271
time: 77.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2800346, upper bound: 70.2971233
time: 78.70 seconds

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

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1436

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 968

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3039359, upper bound: 70.2805577
time: 106.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3086479, upper bound: 70.2758308
time: 81.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 189.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 189.99
Output dim: 4, lower bound: -70.2690263, upper bound: 70.3081271
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 189.99
Output dim: 4, lower bound: -70.2800346, upper bound: 70.2971233
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 189.99
Output dim: 4, lower bound: -70.3039359, upper bound: 70.2805577
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 189.99
Output dim: 4, lower bound: -70.3086479, upper bound: 70.2758308

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

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 973

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1368

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2667971, upper bound: 70.3070643
time: 80.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2679748, upper bound: 70.3059056
time: 104.77 seconds

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

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1416

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1650

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2795361, upper bound: 70.2652110
time: 135.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2542871, upper bound: 70.2961569
time: 94.63 seconds

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

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1629

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3031259, upper bound: 70.2664593
time: 109.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2898143, upper bound: 70.2797447
time: 96.54 seconds

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

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1576

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2761279, upper bound: 70.2751498
time: 90.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3042292, upper bound: 70.2713310
time: 92.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 185.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 185.32
Output dim: 4, lower bound: -70.2667971, upper bound: 70.3070643
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 185.32
Output dim: 4, lower bound: -70.2679748, upper bound: 70.3059056
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 185.32
Output dim: 4, lower bound: -70.2795361, upper bound: 70.2652110
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 185.32
Output dim: 4, lower bound: -70.2542871, upper bound: 70.2961569
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 185.32
Output dim: 4, lower bound: -70.3031259, upper bound: 70.2664593
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 185.32
Output dim: 4, lower bound: -70.2898143, upper bound: 70.2797447
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 185.32
Output dim: 4, lower bound: -70.2761279, upper bound: 70.2751498
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 185.32
Output dim: 4, lower bound: -70.3042292, upper bound: 70.2713310

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

Time for backsubstitution: 2.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1402

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 789

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2667969, upper bound: 70.2952228
time: 80.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2549776, upper bound: 70.3070641
time: 79.76 seconds

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1560

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1387

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2607183, upper bound: 70.2986786
time: 152.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2607183, upper bound: 70.3013344
time: 85.88 seconds

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

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1576

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2750983, upper bound: 70.2645166
time: 188.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2750983, upper bound: 70.2606389
time: 102.11 seconds

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

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 594

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2526616, upper bound: 70.2586981
time: 92.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2158282, upper bound: 70.2946341
time: 98.20 seconds

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

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1700

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 965

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3028800, upper bound: 70.2660045
time: 84.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3026707, upper bound: 70.2662172
time: 94.51 seconds

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

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1681

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2892994, upper bound: 70.2794207
time: 89.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2895074, upper bound: 70.2792220
time: 122.87 seconds

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1754

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 892

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2837503, upper bound: 70.2736904
time: 86.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2837503, upper bound: 70.2546320
time: 98.79 seconds

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

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1705

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2797755, upper bound: 70.2429438
time: 81.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2797755, upper bound: 70.2429438
time: 80.84 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 164.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2667969, upper bound: 70.2952228
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2549776, upper bound: 70.3070641
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2607183, upper bound: 70.2986786
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2607183, upper bound: 70.3013344
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2750983, upper bound: 70.2645166
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2750983, upper bound: 70.2606389
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2526616, upper bound: 70.2586981
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2158282, upper bound: 70.2946341
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.3028800, upper bound: 70.2660045
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.3026707, upper bound: 70.2662172
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2892994, upper bound: 70.2794207
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2895074, upper bound: 70.2792220
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2837503, upper bound: 70.2736904
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2837503, upper bound: 70.2546320
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2797755, upper bound: 70.2429438
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 164.90
Output dim: 4, lower bound: -70.2797755, upper bound: 70.2429438

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 526

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2650984, upper bound: 70.2926748
time: 78.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2642150, upper bound: 70.2935571
time: 109.06 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 984

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 844

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2466905, upper bound: 70.3068760
time: 201.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2466905, upper bound: 70.2988648
time: 99.00 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 737

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2541504, upper bound: 70.2882391
time: 195.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2502150, upper bound: 70.2894220
time: 83.94 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1679

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2592103, upper bound: 70.2859278
time: 74.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2453194, upper bound: 70.2998366
time: 88.00 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1491

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1743

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2572084, upper bound: 70.2640841
time: 100.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2746658, upper bound: 70.2466354
time: 82.16 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 522

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1614

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2696510, upper bound: 70.2601108
time: 86.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2743683, upper bound: 70.2541771
time: 76.42 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1712

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 942

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2522328, upper bound: 70.2572852
time: 87.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2512602, upper bound: 70.2582664
time: 100.84 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 886

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 673

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2038692, upper bound: 70.2900224
time: 88.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2102300, upper bound: 70.2809438
time: 87.96 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1417

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 772

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3026679, upper bound: 70.2633097
time: 88.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3001951, upper bound: 70.2657917
time: 94.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 923

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1003

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2975255, upper bound: 70.2613389
time: 103.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2977670, upper bound: 70.2610892
time: 417.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2608970, upper bound: 70.2512555
time: 86.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2608970, upper bound: 70.2512555
time: 195.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 626

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 969

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2812046, upper bound: 70.2787179
time: 97.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2889097, upper bound: 70.2682618
time: 82.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1715

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 530

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2778017, upper bound: 70.2690439
time: 99.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2791432, upper bound: 70.2676914
time: 130.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1722

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1726

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2234145, upper bound: 70.2540923
time: 86.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.3022171, upper bound: 70.2272603
time: 105.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1757

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1451

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2748835, upper bound: 70.2421312
time: 91.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2789584, upper bound: 70.2420005
time: 86.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 706

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2585654, upper bound: 70.2421890
time: 129.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2790175, upper bound: 70.2217633
time: 105.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 237.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2650984, upper bound: 70.2926748
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2642150, upper bound: 70.2935571
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2466905, upper bound: 70.3068760
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2466905, upper bound: 70.2988648
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2541504, upper bound: 70.2882391
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2502150, upper bound: 70.2894220
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2592103, upper bound: 70.2859278
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2453194, upper bound: 70.2998366
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2572084, upper bound: 70.2640841
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2746658, upper bound: 70.2466354
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2696510, upper bound: 70.2601108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2743683, upper bound: 70.2541771
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2522328, upper bound: 70.2572852
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2512602, upper bound: 70.2582664
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2038692, upper bound: 70.2900224
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2102300, upper bound: 70.2809438
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.3026679, upper bound: 70.2633097
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.3001951, upper bound: 70.2657917
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2975255, upper bound: 70.2613389
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2977670, upper bound: 70.2610892
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2608970, upper bound: 70.2512555
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2608970, upper bound: 70.2512555
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2812046, upper bound: 70.2787179
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2889097, upper bound: 70.2682618
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2778017, upper bound: 70.2690439
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2791432, upper bound: 70.2676914
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2234145, upper bound: 70.2540923
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.3022171, upper bound: 70.2272603
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2748835, upper bound: 70.2421312
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2789584, upper bound: 70.2420005
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2585654, upper bound: 70.2421890
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 237.32
Output dim: 4, lower bound: -70.2790175, upper bound: 70.2217633

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 947

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 798

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2649084, upper bound: 70.2832541
time: 91.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2553581, upper bound: 70.2922734
time: 83.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1624

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1664

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2641501, upper bound: 70.2900390
time: 80.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2606959, upper bound: 70.2934853
time: 88.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1515
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1514
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1526
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1730

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2159196, upper bound: 70.2762307
time: 91.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2159196, upper bound: 70.2762307
time: 91.21 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 184.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 184.88
Output dim: 4, lower bound: -70.2649084, upper bound: 70.2832541
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 184.88
Output dim: 4, lower bound: -70.2553581, upper bound: 70.2922734
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 184.88
Output dim: 4, lower bound: -70.2641501, upper bound: 70.2900390
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 184.88
Output dim: 4, lower bound: -70.2606959, upper bound: 70.2934853
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 184.88
Output dim: 4, lower bound: -70.2159196, upper bound: 70.2762307
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 184.88
Output dim: 4, lower bound: -70.2159196, upper bound: 70.2762307
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2466905, upper bound: 70.2988648
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2541504, upper bound: 70.2882391
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2502150, upper bound: 70.2894220
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2592103, upper bound: 70.2859278
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2453194, upper bound: 70.2998366
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2572084, upper bound: 70.2640841
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2746658, upper bound: 70.2466354
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2696510, upper bound: 70.2601108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2743683, upper bound: 70.2541771
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2522328, upper bound: 70.2572852
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2512602, upper bound: 70.2582664
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2038692, upper bound: 70.2900224
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2102300, upper bound: 70.2809438
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.3026679, upper bound: 70.2633097
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.3001951, upper bound: 70.2657917
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2975255, upper bound: 70.2613389
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2977670, upper bound: 70.2610892
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2608970, upper bound: 70.2512555
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2608970, upper bound: 70.2512555
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2812046, upper bound: 70.2787179
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2889097, upper bound: 70.2682618
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2778017, upper bound: 70.2690439
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2791432, upper bound: 70.2676914
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2234145, upper bound: 70.2540923
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.3022171, upper bound: 70.2272603
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2748835, upper bound: 70.2421312
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2789584, upper bound: 70.2420005
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2585654, upper bound: 70.2421890
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 184.88
Output dim: 4, lower bound: -70.2790175, upper bound: 70.2217633

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 96.57 + 7201.69 = 7298.26 seconds
