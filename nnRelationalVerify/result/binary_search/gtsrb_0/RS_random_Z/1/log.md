## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 85.7987861207
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805)
1: (-67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712)
2: (-57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314)
3: (-74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054)
4: (-73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091)
5: (-69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532)
6: (-100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046)
7: (-84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995)
8: (-101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853)
9: (-72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778)
10: (-96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518)
11: (-90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414)
12: (-104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341)
13: (-112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724)
14: (-160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609)
15: (-80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428)
16: (-99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911)
17: (-157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005)
18: (-99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747)
19: (-64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750)
20: (-69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685)
21: (-84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612)
22: (-88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659)
23: (-67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749)
24: (-88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396)
25: (-77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250)
26: (-109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865)
27: (-88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920)
28: (-69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609)
29: (-90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527)
30: (-89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227)
31: (-87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861)
32: (-100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711)
33: (-133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821)
34: (-117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296)
35: (-108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085)
36: (-111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927)
37: (-152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648)
38: (-133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826)
39: (-149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499)
40: (-115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975)
41: (-105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187)
42: (-76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904)

## BASE Result
execution time: IAR + LP analysis = 2.95 + 153.57 = 156.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 19, lower bound: -89.5643979, upper bound: 89.5643979


# Binary Search by BASE starts (time budget: 17843.48 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=106.16907501220703
rel_dist={19: [-85.9497628281792, 85.94976282669441]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=106.16907501220703
rel_dist={19: [-83.03806944157836, 83.03806944209214]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=106.16907501220703
rel_dist={19: [-84.12058168343383, 84.12058168603599]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=106.16907501220703
rel_dist={19: [-85.09092022017158, 85.0909202198491]}

## Binary Search Result
Binary search time: 751.70 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 17091.78 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1682

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 572

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0304547, upper bound: 88.0222802
time: 175.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0222802, upper bound: 88.0304547
time: 154.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 329.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 329.90
Output dim: 19, lower bound: -88.0304547, upper bound: 88.0222802
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 329.90
Output dim: 19, lower bound: -88.0222802, upper bound: 88.0304547

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1561

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 891

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0191864, upper bound: 88.0193391
time: 193.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0275231, upper bound: 88.0111053
time: 197.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1710

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 532

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0188317, upper bound: 88.0200527
time: 179.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0119102, upper bound: 88.0270937
time: 192.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 374.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 374.28
Output dim: 19, lower bound: -88.0191864, upper bound: 88.0193391
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 374.28
Output dim: 19, lower bound: -88.0275231, upper bound: 88.0111053
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 374.28
Output dim: 19, lower bound: -88.0188317, upper bound: 88.0200527
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 374.28
Output dim: 19, lower bound: -88.0119102, upper bound: 88.0270937

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1558

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 857

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9515832, upper bound: 87.9842729
time: 226.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9835739, upper bound: 87.9523400
time: 175.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1662

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1644

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0263824, upper bound: 88.0107944
time: 204.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0271369, upper bound: 88.0100670
time: 195.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 521

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1791

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0151392, upper bound: 88.0199099
time: 222.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0186827, upper bound: 88.0163045
time: 147.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1582

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1200

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0095635, upper bound: 88.0269064
time: 176.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0095635, upper bound: 88.0247911
time: 198.99 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 377.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 377.42
Output dim: 19, lower bound: -87.9515832, upper bound: 87.9842729
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 377.42
Output dim: 19, lower bound: -87.9835739, upper bound: 87.9523400
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 377.42
Output dim: 19, lower bound: -88.0263824, upper bound: 88.0107944
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 377.42
Output dim: 19, lower bound: -88.0271369, upper bound: 88.0100670
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 377.42
Output dim: 19, lower bound: -88.0151392, upper bound: 88.0199099
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 377.42
Output dim: 19, lower bound: -88.0186827, upper bound: 88.0163045
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 377.42
Output dim: 19, lower bound: -88.0095635, upper bound: 88.0269064
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 377.42
Output dim: 19, lower bound: -88.0095635, upper bound: 88.0247911

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 531

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9514815, upper bound: 87.9804289
time: 163.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9477245, upper bound: 87.9841158
time: 153.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1594

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 885

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9554444, upper bound: 87.9508027
time: 209.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9820463, upper bound: 87.9240807
time: 189.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1027

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9285732, upper bound: 87.9108802
time: 164.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9262554, upper bound: 87.9132864
time: 229.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1623

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1248

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0222680, upper bound: 88.0099944
time: 247.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0270642, upper bound: 88.0051850
time: 175.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 425.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 425.56
Output dim: 19, lower bound: -87.9514815, upper bound: 87.9804289
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 425.56
Output dim: 19, lower bound: -87.9477245, upper bound: 87.9841158
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 425.56
Output dim: 19, lower bound: -87.9554444, upper bound: 87.9508027
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 425.56
Output dim: 19, lower bound: -87.9820463, upper bound: 87.9240807
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 425.56
Output dim: 19, lower bound: -87.9285732, upper bound: 87.9108802
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 425.56
Output dim: 19, lower bound: -87.9262554, upper bound: 87.9132864
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 425.56
Output dim: 19, lower bound: -88.0222680, upper bound: 88.0099944
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 425.56
Output dim: 19, lower bound: -88.0270642, upper bound: 88.0051850
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 425.56
Output dim: 19, lower bound: -88.0151392, upper bound: 88.0199099
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 425.56
Output dim: 19, lower bound: -88.0186827, upper bound: 88.0163045
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 425.56
Output dim: 19, lower bound: -88.0095635, upper bound: 88.0269064
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 425.56
Output dim: 19, lower bound: -88.0095635, upper bound: 88.0247911
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=106.16907501220703
rel_dist={19: [-88.03682671653708, 88.03682671776684]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 623

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.7152694, upper bound: 86.7027750
time: 155.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.7027750, upper bound: 86.7152694
time: 148.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 304.03 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 304.03
Output dim: 19, lower bound: -86.7152694, upper bound: 86.7027750
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 304.03
Output dim: 19, lower bound: -86.7027750, upper bound: 86.7152694

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 923

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1124

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6981670, upper bound: 86.7027459
time: 491.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.7152403, upper bound: 86.6856369
time: 169.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6750422, upper bound: 86.6827642
time: 278.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6703660, upper bound: 86.6873668
time: 179.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 460.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 460.52
Output dim: 19, lower bound: -86.6981670, upper bound: 86.7027459
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 460.52
Output dim: 19, lower bound: -86.7152403, upper bound: 86.6856369
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 460.52
Output dim: 19, lower bound: -86.6750422, upper bound: 86.6827642
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 460.52
Output dim: 19, lower bound: -86.6703660, upper bound: 86.6873668

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 625

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 522

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6850456, upper bound: 86.6965584
time: 176.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6919841, upper bound: 86.7021470
time: 169.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 719

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.7141047, upper bound: 86.6849175
time: 217.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.7020418, upper bound: 86.6844545
time: 175.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6587075, upper bound: 86.6822684
time: 250.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6745608, upper bound: 86.6674701
time: 211.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 549

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6696715, upper bound: 86.6852553
time: 212.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6682597, upper bound: 86.6866879
time: 179.85 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 394.26 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 394.26
Output dim: 19, lower bound: -86.6850456, upper bound: 86.6965584
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 394.26
Output dim: 19, lower bound: -86.6919841, upper bound: 86.7021470
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 394.26
Output dim: 19, lower bound: -86.7141047, upper bound: 86.6849175
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 394.26
Output dim: 19, lower bound: -86.7020418, upper bound: 86.6844545
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 394.26
Output dim: 19, lower bound: -86.6587075, upper bound: 86.6822684
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 394.26
Output dim: 19, lower bound: -86.6745608, upper bound: 86.6674701
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 394.26
Output dim: 19, lower bound: -86.6696715, upper bound: 86.6852553
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 394.26
Output dim: 19, lower bound: -86.6682597, upper bound: 86.6866879

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1232

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1667

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6941565, upper bound: 86.6887562
time: 190.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6899059, upper bound: 86.6931337
time: 200.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1002

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6775086, upper bound: 86.6881307
time: 201.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6778889, upper bound: 86.6877717
time: 149.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1588

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6980907, upper bound: 86.6842689
time: 156.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6980907, upper bound: 86.6681728
time: 786.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 945.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 945.07
Output dim: 19, lower bound: -86.6941565, upper bound: 86.6887562
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 945.07
Output dim: 19, lower bound: -86.6899059, upper bound: 86.6931337
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 945.07
Output dim: 19, lower bound: -86.6775086, upper bound: 86.6881307
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 945.07
Output dim: 19, lower bound: -86.6778889, upper bound: 86.6877717
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 945.07
Output dim: 19, lower bound: -86.6980907, upper bound: 86.6842689
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 945.07
Output dim: 19, lower bound: -86.6980907, upper bound: 86.6681728
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 945.07
Output dim: 19, lower bound: -86.7020418, upper bound: 86.6844545
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 945.07
Output dim: 19, lower bound: -86.6587075, upper bound: 86.6822684
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 945.07
Output dim: 19, lower bound: -86.6745608, upper bound: 86.6674701
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 945.07
Output dim: 19, lower bound: -86.6696715, upper bound: 86.6852553
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 945.07
Output dim: 19, lower bound: -86.6682597, upper bound: 86.6866879
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=106.16907501220703
rel_dist={19: [-86.72003761785648, 86.72003761772815]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9490461, upper bound: 85.9490581
time: 167.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9490581, upper bound: 85.9490461
time: 135.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 303.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 303.19
Output dim: 19, lower bound: -85.9490461, upper bound: 85.9490581
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 303.19
Output dim: 19, lower bound: -85.9490581, upper bound: 85.9490461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 775

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9488823, upper bound: 85.9194364
time: 166.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9194214, upper bound: 85.9488944
time: 179.15 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 576

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1201

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9488153, upper bound: 85.9447581
time: 183.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9447692, upper bound: 85.9488037
time: 157.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 343.51 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 343.51
Output dim: 19, lower bound: -85.9488823, upper bound: 85.9194364
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 343.51
Output dim: 19, lower bound: -85.9194214, upper bound: 85.9488944
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 343.51
Output dim: 19, lower bound: -85.9488153, upper bound: 85.9447581
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 343.51
Output dim: 19, lower bound: -85.9447692, upper bound: 85.9488037

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 996

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9200578, upper bound: 85.9190680
time: 155.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9485137, upper bound: 85.8905892
time: 198.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 674

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1704

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9059043, upper bound: 85.9359179
time: 160.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9063587, upper bound: 85.9354661
time: 146.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1620

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 603

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9471278, upper bound: 85.9426330
time: 188.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9466800, upper bound: 85.9430955
time: 169.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1660

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1614

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9420825, upper bound: 85.9485986
time: 180.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9420825, upper bound: 85.9461493
time: 185.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 368.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 368.65
Output dim: 19, lower bound: -85.9200578, upper bound: 85.9190680
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 368.65
Output dim: 19, lower bound: -85.9485137, upper bound: 85.8905892
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 368.65
Output dim: 19, lower bound: -85.9059043, upper bound: 85.9359179
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 368.65
Output dim: 19, lower bound: -85.9063587, upper bound: 85.9354661
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 368.65
Output dim: 19, lower bound: -85.9471278, upper bound: 85.9426330
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 368.65
Output dim: 19, lower bound: -85.9466800, upper bound: 85.9430955
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 368.65
Output dim: 19, lower bound: -85.9420825, upper bound: 85.9485986
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 368.65
Output dim: 19, lower bound: -85.9420825, upper bound: 85.9461493

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1716

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1683

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9148543, upper bound: 85.9052930
time: 252.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9064308, upper bound: 85.9138366
time: 237.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 979

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 517

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9483330, upper bound: 85.8784411
time: 204.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9363533, upper bound: 85.8904054
time: 292.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1553

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 983

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9044825, upper bound: 85.8817898
time: 204.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8517997, upper bound: 85.9345005
time: 147.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -132.3192139, 77.7801666, -132.3192139, 77.7801666, -210.0993652, 210.0993805
1: -67.4001694, 56.0716057, -67.4001694, 56.0716057, -123.4717712, 123.4717712
2: -57.1037750, 60.3849640, -57.1037750, 60.3849640, -117.4887238, 117.4887314
3: -74.0167999, 70.1775055, -74.0167999, 70.1775055, -144.1943054, 144.1943054
4: -73.7660065, 69.7702026, -73.7660065, 69.7702026, -143.5362091, 143.5362091
5: -69.9592438, 72.2815094, -69.9592438, 72.2815094, -142.2407532, 142.2407532
6: -100.3566895, 73.2836304, -100.3566895, 73.2836304, -173.6403198, 173.6403046
7: -84.5409851, 67.6447296, -84.5409851, 67.6447296, -152.1856842, 152.1856995
8: -101.7626190, 87.1766663, -101.7626190, 87.1766663, -188.9392853, 188.9392853
9: -72.0918808, 72.4928894, -72.0918808, 72.4928894, -144.5847778, 144.5847778
10: -96.2133179, 87.9035568, -96.2133179, 87.9035568, -184.1168823, 184.1168518
11: -90.4185638, 58.8282814, -90.4185638, 58.8282814, -149.2468414, 149.2468414
12: -104.8601608, 89.7597733, -104.8601608, 89.7597733, -194.6199341, 194.6199341
13: -112.2070389, 99.0528717, -112.2070389, 99.0528717, -211.2598877, 211.2598724
14: -160.5371246, 76.6271362, -160.5371246, 76.6271362, -237.1642609, 237.1642609
15: -80.2436066, 66.7409363, -80.2436066, 66.7409363, -146.9845428, 146.9845428
16: -99.5209122, 71.6536789, -99.5209122, 71.6536789, -171.1745911, 171.1745911
17: -157.9295349, 74.5834656, -157.9295349, 74.5834656, -232.5130005, 232.5130005
18: -99.7333984, 88.3293762, -99.7333984, 88.3293762, -188.0627747, 188.0627747
19: -64.9778824, 41.1911926, -64.9778824, 41.1911926, -106.1690750, 106.1690750
20: -69.4878082, 53.1792641, -69.4878082, 53.1792641, -122.6670685, 122.6670685
21: -84.8423920, 53.8115883, -84.8423920, 53.8115883, -138.6539612, 138.6539612
22: -88.9188080, 52.7112503, -88.9188080, 52.7112503, -141.6300659, 141.6300659
23: -67.5519104, 57.7160683, -67.5519104, 57.7160683, -125.2679749, 125.2679749
24: -88.2665405, 66.0253143, -88.2665405, 66.0253143, -154.2918396, 154.2918396
25: -77.9071198, 60.1597977, -77.9071198, 60.1597977, -138.0669098, 138.0669250
26: -109.4059296, 90.7020569, -109.4059296, 90.7020569, -200.1079865, 200.1079865
27: -88.0128784, 66.4377060, -88.0128784, 66.4377060, -154.4505920, 154.4505920
28: -69.1515427, 62.5071144, -69.1515427, 62.5071144, -131.6586609, 131.6586609
29: -90.2933121, 47.3719406, -90.2933121, 47.3719406, -137.6652527, 137.6652527
30: -89.2297592, 74.5734558, -89.2297592, 74.5734558, -163.8032074, 163.8032227
31: -87.1362762, 64.1070099, -87.1362762, 64.1070099, -151.2432556, 151.2432861
32: -100.1435471, 64.6631241, -100.1435471, 64.6631241, -164.8066711, 164.8066711
33: -133.0400085, 83.3203735, -133.0400085, 83.3203735, -216.3603668, 216.3603821
34: -117.4808350, 62.7833061, -117.4808350, 62.7833061, -180.2641144, 180.2641296
35: -108.4727325, 71.5159760, -108.4727325, 71.5159760, -179.9887085, 179.9887085
36: -111.2083664, 70.2048340, -111.2083664, 70.2048340, -181.4131775, 181.4131927
37: -152.0579224, 71.0853577, -152.0579224, 71.0853577, -223.1432800, 223.1432648
38: -133.3066559, 84.8792343, -133.3066559, 84.8792343, -218.1858826, 218.1858826
39: -149.1947632, 87.7907867, -149.1947632, 87.7907867, -236.9855347, 236.9855499
40: -115.5572052, 66.5709076, -115.5572052, 66.5709076, -182.1280975, 182.1280975
41: -105.0567169, 75.5343094, -105.0567169, 75.5343094, -180.5910187, 180.5910187
42: -76.4902802, 57.2786026, -76.4902802, 57.2786026, -133.7688751, 133.7688904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=526, inp2_unstable=526, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 890
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 907
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 889
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1184
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1026

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8906583, upper bound: 85.9354374
time: 166.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9063300, upper bound: 85.9197765
time: 154.14 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 323.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 323.14
Output dim: 19, lower bound: -85.9148543, upper bound: 85.9052930
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 323.14
Output dim: 19, lower bound: -85.9064308, upper bound: 85.9138366
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 323.14
Output dim: 19, lower bound: -85.9483330, upper bound: 85.8784411
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 323.14
Output dim: 19, lower bound: -85.9363533, upper bound: 85.8904054
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 323.14
Output dim: 19, lower bound: -85.9044825, upper bound: 85.8817898
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 323.14
Output dim: 19, lower bound: -85.8517997, upper bound: 85.9345005
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 323.14
Output dim: 19, lower bound: -85.8906583, upper bound: 85.9354374
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 323.14
Output dim: 19, lower bound: -85.9063300, upper bound: 85.9197765
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 323.14
Output dim: 19, lower bound: -85.9471278, upper bound: 85.9426330
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 323.14
Output dim: 19, lower bound: -85.9466800, upper bound: 85.9430955
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 323.14
Output dim: 19, lower bound: -85.9420825, upper bound: 85.9485986
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 323.14
Output dim: 19, lower bound: -85.9420825, upper bound: 85.9461493
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=106.16907501220703
rel_dist={19: [-85.9497628281792, 85.94976282669441]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 13504.30 seconds
