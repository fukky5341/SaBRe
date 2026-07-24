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
execution time: IAR + LP analysis = 2.96 + 152.60 = 155.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 19, lower bound: -89.5643979, upper bound: 89.5643979


# Binary Search by BASE starts (time budget: 17844.44 seconds, max iter: 100)

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
Binary search time: 749.87 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 17094.57 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 627

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0279601, upper bound: 88.0057231
time: 157.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0279601, upper bound: 88.0279600
time: 429.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 587.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 587.29
Output dim: 19, lower bound: -88.0279601, upper bound: 88.0057231
IS_A2, status: Status.UNKNOWN, split count: 1, time: 587.29
Output dim: 19, lower bound: -88.0279601, upper bound: 88.0279600

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -132.0718079, 77.6194534, -132.2783661, 77.7668152, -209.8386230, 209.8978119
1: -67.3108826, 55.9463425, -67.3841171, 56.0601616, -123.3710403, 123.3304596
2: -56.9860611, 60.2972565, -57.0822029, 60.3770676, -117.3631210, 117.3794403
3: -73.7489014, 69.9801331, -73.9650116, 70.1673965, -143.9162903, 143.9451294
4: -73.5577011, 69.5926971, -73.7270355, 69.7586899, -143.3163757, 143.3197327
5: -69.7194443, 72.0991135, -69.9143295, 72.2715149, -141.9909668, 142.0134430
6: -100.2175293, 73.2617950, -100.3386383, 73.2682343, -173.4857635, 173.6004333
7: -84.3854980, 67.5366058, -84.5110626, 67.6353531, -152.0208435, 152.0476685
8: -101.5638351, 87.0094604, -101.7246857, 87.1630096, -188.7268372, 188.7341461
9: -71.8535538, 72.2626801, -72.0475616, 72.4814148, -144.3349609, 144.3102417
10: -95.8962021, 87.6484070, -96.1568146, 87.8892593, -183.7854614, 183.8052063
11: -90.1747513, 58.6748085, -90.3947449, 58.8001480, -148.9748840, 149.0695496
12: -104.7622833, 89.6147766, -104.8466568, 89.7356720, -194.4979248, 194.4614258
13: -111.9980164, 98.8696289, -112.1686935, 99.0328445, -211.0308075, 211.0383301
14: -160.3263550, 76.5023117, -160.5036621, 76.6068954, -236.9331970, 237.0059509
15: -80.0462723, 66.5190582, -80.2089157, 66.7280960, -146.7743530, 146.7279663
16: -99.3466339, 71.4986572, -99.4939804, 71.6406555, -170.9872742, 170.9926147
17: -157.6744843, 74.4459991, -157.8862457, 74.5636063, -232.2380981, 232.3322449
18: -99.5259781, 88.1391449, -99.7098160, 88.2916565, -187.8176270, 187.8489380
19: -64.7676392, 41.0882874, -64.9615631, 41.1709671, -105.9386063, 106.0498505
20: -69.3254242, 53.0810738, -69.4728699, 53.1628456, -122.4882660, 122.5539398
21: -84.6373825, 53.6940536, -84.8222351, 53.7907753, -138.4281464, 138.5162964
22: -88.7699509, 52.6141624, -88.9019928, 52.6939812, -141.4639282, 141.5161438
23: -67.2905960, 57.4670525, -67.5355682, 57.6671867, -124.9577789, 125.0026245
24: -88.0033340, 65.7506714, -88.2523041, 65.9708633, -153.9741821, 154.0029755
25: -77.6770020, 59.9489326, -77.8921738, 60.1197624, -137.7967529, 137.8410950
26: -109.2186737, 90.5307465, -109.3848801, 90.6708145, -199.8894653, 199.9156189
27: -87.8052216, 66.1688385, -87.9996948, 66.3854523, -154.1906738, 154.1685333
28: -68.9475250, 62.3041115, -69.1392822, 62.4699059, -131.4174347, 131.4433594
29: -90.1294479, 47.2531052, -90.2751770, 47.3498039, -137.4792480, 137.5282745
30: -89.0129395, 74.3575745, -89.2144394, 74.5357056, -163.5486450, 163.5720215
31: -86.8671341, 63.9941406, -87.1153259, 64.0858154, -150.9529419, 151.1094666
32: -100.0142975, 64.6272888, -100.1258621, 64.6494293, -164.6637268, 164.7531433
33: -132.8480835, 83.2524719, -133.0199585, 83.3062286, -216.1543121, 216.2724304
34: -117.3650360, 62.7112541, -117.4655685, 62.7676010, -180.1326294, 180.1768188
35: -108.3294449, 71.3791428, -108.4588318, 71.4892731, -179.8187256, 179.8379822
36: -111.0593414, 70.0432739, -111.1955872, 70.1742249, -181.2335663, 181.2388611
37: -151.7686462, 70.8333969, -152.0333252, 71.0379181, -222.8065643, 222.8667297
38: -133.1062622, 84.6801224, -133.2874298, 84.8414688, -217.9477234, 217.9675598
39: -148.9757080, 87.6772079, -149.1704102, 87.7692261, -236.7449341, 236.8476257
40: -115.3540039, 66.5429535, -115.5268250, 66.5547256, -181.9087219, 182.0697632
41: -104.9207230, 75.4301758, -105.0417709, 75.5164719, -180.4371948, 180.4719543
42: -76.3620453, 57.2464676, -76.4735260, 57.2675171, -133.6295624, 133.7199860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=526, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=718, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9240491, upper bound: 87.9755110
time: 163.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9240491, upper bound: 87.9481293
time: 162.15 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -132.3091125, 77.7714996, -132.3192139, 77.7801666, -210.0892639, 210.0907135
1: -67.3951111, 56.0654373, -67.4001694, 56.0716057, -123.4667206, 123.4656067
2: -57.0985336, 60.3796921, -57.1037750, 60.3849640, -117.4834976, 117.4834518
3: -74.0066071, 70.1723938, -74.0167999, 70.1775055, -144.1841125, 144.1891785
4: -73.7588806, 69.7627411, -73.7660065, 69.7702026, -143.5290680, 143.5287476
5: -69.9503098, 72.2761917, -69.9592438, 72.2815094, -142.2317963, 142.2354431
6: -100.3205719, 73.2786713, -100.3566895, 73.2836304, -173.6041870, 173.6353607
7: -84.5334396, 67.6393738, -84.5409851, 67.6447296, -152.1781616, 152.1803589
8: -101.7537994, 87.1708069, -101.7626190, 87.1766663, -188.9304657, 188.9334106
9: -72.0829010, 72.4865265, -72.0918808, 72.4928894, -144.5757751, 144.5783997
10: -96.2020493, 87.8952942, -96.2133179, 87.9035568, -184.1056061, 184.1085815
11: -90.4096069, 58.8208885, -90.4185638, 58.8282814, -149.2378845, 149.2394562
12: -104.8503036, 89.7528305, -104.8601608, 89.7597733, -194.6100769, 194.6129913
13: -112.1929169, 99.0453033, -112.2070389, 99.0528717, -211.2457886, 211.2523041
14: -160.5264740, 76.6091156, -160.5371246, 76.6271362, -237.1535950, 237.1462402
15: -80.2358475, 66.7335510, -80.2436066, 66.7409363, -146.9767761, 146.9771423
16: -99.5118332, 71.6465149, -99.5209122, 71.6536789, -171.1654968, 171.1674194
17: -157.9189301, 74.5764618, -157.9295349, 74.5834656, -232.5023804, 232.5059967
18: -99.7268982, 88.3162308, -99.7333984, 88.3293762, -188.0562744, 188.0496216
19: -64.9723663, 41.1811447, -64.9778824, 41.1911926, -106.1635437, 106.1590195
20: -69.4813843, 53.1743393, -69.4878082, 53.1792641, -122.6606445, 122.6621399
21: -84.8355408, 53.7853394, -84.8423920, 53.8115883, -138.6471100, 138.6277313
22: -88.9120178, 52.7000275, -88.9188080, 52.7112503, -141.6232605, 141.6188354
23: -67.5456543, 57.7058563, -67.5519104, 57.7160683, -125.2617188, 125.2577667
24: -88.2593384, 66.0144196, -88.2665405, 66.0253143, -154.2846375, 154.2809448
25: -77.9002457, 60.1505051, -77.9071198, 60.1597977, -138.0600433, 138.0576172
26: -109.3970261, 90.6831818, -109.4059296, 90.7020569, -200.0990906, 200.0891113
27: -88.0051117, 66.4275208, -88.0128784, 66.4377060, -154.4428101, 154.4403992
28: -69.1465912, 62.4998474, -69.1515427, 62.5071144, -131.6537018, 131.6513977
29: -90.2856140, 47.3662109, -90.2933121, 47.3719406, -137.6575623, 137.6595154
30: -89.2205734, 74.5660553, -89.2297592, 74.5734558, -163.7940216, 163.7958069
31: -87.1298294, 64.0869904, -87.1362762, 64.1070099, -151.2368164, 151.2232666
32: -100.1118546, 64.6576691, -100.1435471, 64.6631241, -164.7749786, 164.8012085
33: -133.0290833, 83.3148499, -133.0400085, 83.3203735, -216.3494568, 216.3548584
34: -117.4673386, 62.7784157, -117.4808350, 62.7833061, -180.2506409, 180.2592316
35: -108.4649048, 71.5093231, -108.4727325, 71.5159760, -179.9808807, 179.9820557
36: -111.1997452, 70.1994781, -111.2083664, 70.2048340, -181.4045715, 181.4078369
37: -152.0449677, 71.0758362, -152.0579224, 71.0853577, -223.1302948, 223.1337280
38: -133.2972717, 84.8709869, -133.3066559, 84.8792343, -218.1765137, 218.1776428
39: -149.1843262, 87.7836761, -149.1947632, 87.7907867, -236.9750977, 236.9784393
40: -115.5243454, 66.5658417, -115.5572052, 66.5709076, -182.0952454, 182.1230316
41: -105.0462570, 75.5291519, -105.0567169, 75.5343094, -180.5805511, 180.5858612
42: -76.4584351, 57.2741852, -76.4902802, 57.2786026, -133.7370300, 133.7644653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=526, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 627

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0057231, upper bound: 88.0279601
time: 173.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -88.0057231, upper bound: 88.0279601
time: 215.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 391.35 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 391.35
Output dim: 19, lower bound: -87.9240491, upper bound: 87.9755110
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 391.35
Output dim: 19, lower bound: -87.9240491, upper bound: 87.9481293
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 391.35
Output dim: 19, lower bound: -88.0057231, upper bound: 88.0279601
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 391.35
Output dim: 19, lower bound: -88.0057231, upper bound: 88.0279601

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -132.0685577, 77.6144409, -132.2172852, 77.6679916, -209.7365417, 209.8317261
1: -67.3095551, 55.9434357, -67.3591003, 56.0022354, -123.3117676, 123.3025360
2: -56.9848709, 60.2937622, -57.0594635, 60.3134613, -117.2983322, 117.3532257
3: -73.7470093, 69.9781647, -73.9301605, 70.1300049, -143.8770142, 143.9083252
4: -73.5559387, 69.5867767, -73.6966476, 69.6423492, -143.1982880, 143.2834167
5: -69.7182465, 72.0951080, -69.8910217, 72.1959000, -141.9141388, 141.9861145
6: -100.2094116, 73.2601471, -100.1743393, 73.2365952, -173.4460144, 173.4344788
7: -84.3838959, 67.5322342, -84.4799881, 67.5514908, -151.9353790, 152.0122223
8: -101.5625458, 87.0040359, -101.7006760, 87.0574265, -188.6199646, 188.7047119
9: -71.8470840, 72.2609253, -71.9281845, 72.4481354, -144.2952271, 144.1891022
10: -95.8926010, 87.6453018, -96.0882263, 87.8306732, -183.7232666, 183.7335205
11: -90.1714554, 58.6705437, -90.3310928, 58.7218475, -148.8933105, 149.0016174
12: -104.7505341, 89.6122131, -104.6136932, 89.6855087, -194.4360046, 194.2259064
13: -111.9851379, 98.8672485, -111.9288483, 98.9859314, -210.9710541, 210.7960968
14: -160.3222656, 76.4910431, -160.4279785, 76.3943176, -236.7165833, 236.9190063
15: -80.0438538, 66.5167236, -80.1614227, 66.6823730, -146.7262115, 146.6781311
16: -99.3433914, 71.4949493, -99.4319000, 71.5697556, -170.9131317, 170.9268494
17: -157.6708984, 74.4329147, -157.8215485, 74.3085632, -231.9794617, 232.2544556
18: -99.5228958, 88.1254425, -99.6506119, 88.0318604, -187.5547485, 187.7760620
19: -64.7660828, 41.0835800, -64.9321899, 41.0789337, -105.8450089, 106.0157471
20: -69.3232117, 53.0771179, -69.4292984, 53.0856285, -122.4088440, 122.5064011
21: -84.6350708, 53.6885376, -84.7784653, 53.6853180, -138.3203888, 138.4670105
22: -88.7667542, 52.6103630, -88.8401108, 52.6207428, -141.3874969, 141.4504700
23: -67.2888489, 57.4620323, -67.5023193, 57.5724258, -124.8612671, 124.9643555
24: -88.0000610, 65.7405701, -88.1886139, 65.7757950, -153.7758484, 153.9291840
25: -77.6748428, 59.9418182, -77.8516235, 59.9841156, -137.6589508, 137.7934418
26: -109.2149048, 90.5252914, -109.3118668, 90.5685272, -199.7834167, 199.8371582
27: -87.8018570, 66.1599121, -87.9337006, 66.2154694, -154.0173187, 154.0936127
28: -68.9456863, 62.2985802, -69.1045761, 62.3605690, -131.3062439, 131.4031525
29: -90.1256943, 47.2500572, -90.2033920, 47.2974968, -137.4231873, 137.4534454
30: -89.0098267, 74.3484497, -89.1549530, 74.3686218, -163.3784180, 163.5033875
31: -86.8647766, 63.9866180, -87.0704498, 63.9421501, -150.8069153, 151.0570679
32: -99.9995117, 64.6249542, -99.8385544, 64.6052322, -164.6047363, 164.4635010
33: -132.8404846, 83.2508469, -132.8795471, 83.2757339, -216.1162109, 216.1304016
34: -117.3609390, 62.7100449, -117.3870468, 62.7456551, -180.1065979, 180.0970764
35: -108.3244019, 71.3782654, -108.3663101, 71.4724121, -179.7968140, 179.7445679
36: -111.0519104, 70.0425186, -111.0504532, 70.1598434, -181.2117615, 181.0929718
37: -151.7612305, 70.8319931, -151.8879089, 71.0106354, -222.7718506, 222.7198944
38: -133.0999603, 84.6782532, -133.1656189, 84.8050537, -217.9050140, 217.8438416
39: -148.9654388, 87.6761017, -148.9807434, 87.7481537, -236.7135925, 236.6568451
40: -115.3461380, 66.5417480, -115.3705673, 66.5324326, -181.8785706, 181.9123230
41: -104.9106369, 75.4284668, -104.8451385, 75.4836578, -180.3942871, 180.2735901
42: -76.3510895, 57.2445183, -76.2540131, 57.2298317, -133.5809174, 133.4985352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=717, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.8844043, upper bound: 87.8761115
time: 186.44 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.8271395, upper bound: 87.8785702
time: 189.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -132.0668488, 77.6076965, -132.3494873, 77.7680664, -209.8349152, 209.9571838
1: -67.3089600, 55.9410515, -67.4256363, 56.0731049, -123.3820648, 123.3666840
2: -56.9844933, 60.2931328, -57.1432724, 60.4062080, -117.3907013, 117.4364014
3: -73.7452393, 69.9765549, -74.0133362, 70.2088928, -143.9541168, 143.9898987
4: -73.5546112, 69.5817566, -73.8189545, 69.7679291, -143.3225403, 143.4007111
5: -69.7177887, 72.0937119, -69.9980469, 72.3122559, -142.0300446, 142.0917664
6: -100.2113571, 73.2599030, -100.3635254, 73.4006195, -173.6119690, 173.6234131
7: -84.3838348, 67.5321274, -84.6043854, 67.6517105, -152.0355377, 152.1365051
8: -101.5622559, 87.0038910, -101.8106613, 87.1875916, -188.7498474, 188.8145447
9: -71.8477936, 72.2602234, -72.0805511, 72.6313477, -144.4791260, 144.3407593
10: -95.8877869, 87.6451111, -96.1897964, 87.9525375, -183.8403320, 183.8348999
11: -90.1721573, 58.6692238, -90.5614090, 58.8269653, -148.9990997, 149.2306213
12: -104.7563019, 89.6124878, -104.8797226, 89.9492035, -194.7054901, 194.4922028
13: -111.9913940, 98.8662872, -112.2106247, 99.3356323, -211.3270264, 211.0769043
14: -160.3211975, 76.4963684, -160.6827087, 76.6038971, -236.9250946, 237.1790619
15: -80.0417633, 66.5170593, -80.2494965, 66.7833633, -146.8251038, 146.7665405
16: -99.3434753, 71.4939270, -99.6601715, 71.6711349, -171.0145874, 171.1540985
17: -157.6697235, 74.4373474, -158.1734619, 74.5707703, -232.2404938, 232.6108093
18: -99.5206146, 88.1257095, -100.0197296, 88.3005371, -187.8211517, 188.1454468
19: -64.7652893, 41.0859871, -65.0986786, 41.1881027, -105.9533920, 106.1846619
20: -69.3233185, 53.0784492, -69.5475769, 53.1744537, -122.4977722, 122.6260223
21: -84.6348572, 53.6899376, -84.9595337, 53.8004341, -138.4352722, 138.6494598
22: -88.7665176, 52.6091118, -89.0296631, 52.7053909, -141.4718933, 141.6387634
23: -67.2887878, 57.4629631, -67.6565247, 57.6899185, -124.9787064, 125.1194687
24: -87.9996567, 65.7462311, -88.4730225, 65.9989929, -153.9986420, 154.2192535
25: -77.6743622, 59.9447441, -77.9869995, 60.1355743, -137.8099365, 137.9317474
26: -109.2147369, 90.5211029, -109.5834656, 90.6751938, -199.8899231, 200.1045685
27: -87.8014832, 66.1646423, -88.2139282, 66.4074860, -154.2089539, 154.3785706
28: -68.9456635, 62.3007240, -69.2665558, 62.4911804, -131.4368134, 131.5672760
29: -90.1255188, 47.2493973, -90.3726807, 47.3672142, -137.4927368, 137.6220703
30: -89.0103149, 74.3511429, -89.4113464, 74.5618057, -163.5721130, 163.7624817
31: -86.8636169, 63.9906883, -87.2787781, 64.1105576, -150.9741821, 151.2694702
32: -100.0066299, 64.6246796, -100.1475906, 64.8700714, -164.8767090, 164.7722626
33: -132.8412323, 83.2507019, -133.0585480, 83.4692383, -216.3104706, 216.3092346
34: -117.3614120, 62.7085953, -117.5117493, 62.8020592, -180.1634674, 180.2203369
35: -108.3243561, 71.3779984, -108.4916382, 71.5482788, -179.8726349, 179.8696289
36: -111.0513611, 70.0424957, -111.2109985, 70.2982864, -181.3496399, 181.2534790
37: -151.7593689, 70.8320618, -152.0731201, 71.1194763, -222.8788452, 222.9051514
38: -133.0979919, 84.6785049, -133.3184052, 84.9628830, -218.0608826, 217.9968872
39: -148.9634094, 87.6763306, -149.1821747, 87.9667358, -236.9301453, 236.8584900
40: -115.3438416, 66.5416489, -115.5517120, 66.6800385, -182.0238800, 182.0933533
41: -104.9155731, 75.4286804, -105.0671768, 75.6520462, -180.5676270, 180.4958496
42: -76.3562317, 57.2437363, -76.4891815, 57.5077477, -133.8639679, 133.7329102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=717, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9286216, upper bound: 87.8463147
time: 222.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.8271395, upper bound: 87.8486294
time: 161.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -132.3091125, 77.7714996, -132.0718079, 77.6194534, -209.9285583, 209.8433075
1: -67.3951111, 56.0654373, -67.3108826, 55.9463425, -123.3414536, 123.3763123
2: -57.0985336, 60.3796921, -56.9860611, 60.2972565, -117.3957901, 117.3657303
3: -74.0066071, 70.1723938, -73.7489014, 69.9801331, -143.9867401, 143.9212952
4: -73.7588806, 69.7627411, -73.5577011, 69.5926971, -143.3515778, 143.3204346
5: -69.9503098, 72.2761917, -69.7194443, 72.0991135, -142.0494232, 141.9956360
6: -100.3205719, 73.2786713, -100.2175293, 73.2617950, -173.5823517, 173.4962006
7: -84.5334396, 67.6393738, -84.3854980, 67.5366058, -152.0700378, 152.0248718
8: -101.7537994, 87.1708069, -101.5638351, 87.0094604, -188.7632599, 188.7346497
9: -72.0829010, 72.4865265, -71.8535538, 72.2626801, -144.3455811, 144.3400726
10: -96.2020493, 87.8952942, -95.8962021, 87.6484070, -183.8504639, 183.7915039
11: -90.4096069, 58.8208885, -90.1747513, 58.6748085, -149.0844116, 148.9956360
12: -104.8503036, 89.7528305, -104.7622833, 89.6147766, -194.4650879, 194.5151062
13: -112.1929169, 99.0453033, -111.9980164, 98.8696289, -211.0625458, 211.0433044
14: -160.5264740, 76.6091156, -160.3263550, 76.5023117, -237.0287628, 236.9354401
15: -80.2358475, 66.7335510, -80.0462723, 66.5190582, -146.7548828, 146.7798157
16: -99.5118332, 71.6465149, -99.3466339, 71.4986572, -171.0104980, 170.9931488
17: -157.9189301, 74.5764618, -157.6744843, 74.4459991, -232.3649292, 232.2509308
18: -99.7268982, 88.3162308, -99.5259781, 88.1391449, -187.8660431, 187.8422089
19: -64.9723663, 41.1811447, -64.7676392, 41.0882874, -106.0606537, 105.9487839
20: -69.4813843, 53.1743393, -69.3254242, 53.0810738, -122.5624542, 122.4997635
21: -84.8355408, 53.7853394, -84.6373825, 53.6940536, -138.5296021, 138.4227295
22: -88.9120178, 52.7000275, -88.7699509, 52.6141624, -141.5261688, 141.4699707
23: -67.5456543, 57.7058563, -67.2905960, 57.4670525, -125.0127106, 124.9964523
24: -88.2593384, 66.0144196, -88.0033340, 65.7506714, -154.0099945, 154.0177612
25: -77.9002457, 60.1505051, -77.6770020, 59.9489326, -137.8491821, 137.8275146
26: -109.3970261, 90.6831818, -109.2186737, 90.5307465, -199.9277649, 199.9018555
27: -88.0051117, 66.4275208, -87.8052216, 66.1688385, -154.1739502, 154.2327423
28: -69.1465912, 62.4998474, -68.9475250, 62.3041115, -131.4506989, 131.4473572
29: -90.2856140, 47.3662109, -90.1294479, 47.2531052, -137.5387268, 137.4956360
30: -89.2205734, 74.5660553, -89.0129395, 74.3575745, -163.5781555, 163.5789795
31: -87.1298294, 64.0869904, -86.8671341, 63.9941406, -151.1239624, 150.9541016
32: -100.1118546, 64.6576691, -100.0142975, 64.6272888, -164.7391357, 164.6719666
33: -133.0290833, 83.3148499, -132.8480835, 83.2524719, -216.2815399, 216.1629333
34: -117.4673386, 62.7784157, -117.3650360, 62.7112541, -180.1785889, 180.1434479
35: -108.4649048, 71.5093231, -108.3294449, 71.3791428, -179.8440247, 179.8387756
36: -111.1997452, 70.1994781, -111.0593414, 70.0432739, -181.2430115, 181.2588196
37: -152.0449677, 71.0758362, -151.7686462, 70.8333969, -222.8783112, 222.8444824
38: -133.2972717, 84.8709869, -133.1062622, 84.6801224, -217.9773865, 217.9772491
39: -149.1843262, 87.7836761, -148.9757080, 87.6772079, -236.8615265, 236.7593842
40: -115.5243454, 66.5658417, -115.3540039, 66.5429535, -182.0672913, 181.9198456
41: -105.0462570, 75.5291519, -104.9207230, 75.4301758, -180.4764404, 180.4498596
42: -76.4584351, 57.2741852, -76.3620453, 57.2464676, -133.7048950, 133.6362305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=717, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9755110, upper bound: 87.9240490
time: 161.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9481293, upper bound: 87.9701951
time: 235.20 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -132.3091125, 77.7714996, -132.3091125, 77.7714996, -210.0806122, 210.0805969
1: -67.3951111, 56.0654373, -67.3951111, 56.0654373, -123.4605255, 123.4605408
2: -57.0985336, 60.3796921, -57.0985336, 60.3796921, -117.4782257, 117.4782257
3: -74.0066071, 70.1723938, -74.0066071, 70.1723938, -144.1790009, 144.1790009
4: -73.7588806, 69.7627411, -73.7588806, 69.7627411, -143.5216217, 143.5216217
5: -69.9503098, 72.2761917, -69.9503098, 72.2761917, -142.2265015, 142.2265015
6: -100.3205719, 73.2786713, -100.3205719, 73.2786713, -173.5992279, 173.5992432
7: -84.5334396, 67.6393738, -84.5334396, 67.6393738, -152.1728058, 152.1728058
8: -101.7537994, 87.1708069, -101.7537994, 87.1708069, -188.9245911, 188.9245911
9: -72.0829010, 72.4865265, -72.0829010, 72.4865265, -144.5694275, 144.5694275
10: -96.2020493, 87.8952942, -96.2020493, 87.8952942, -184.0973511, 184.0973511
11: -90.4096069, 58.8208885, -90.4096069, 58.8208885, -149.2304840, 149.2304993
12: -104.8503036, 89.7528305, -104.8503036, 89.7528305, -194.6031342, 194.6031342
13: -112.1929169, 99.0453033, -112.1929169, 99.0453033, -211.2382202, 211.2382202
14: -160.5264740, 76.6091156, -160.5264740, 76.6091156, -237.1355896, 237.1355896
15: -80.2358475, 66.7335510, -80.2358475, 66.7335510, -146.9693756, 146.9693909
16: -99.5118332, 71.6465149, -99.5118332, 71.6465149, -171.1583252, 171.1583252
17: -157.9189301, 74.5764618, -157.9189301, 74.5764618, -232.4953918, 232.4953918
18: -99.7268982, 88.3162308, -99.7268982, 88.3162308, -188.0431213, 188.0431213
19: -64.9723663, 41.1811447, -64.9723663, 41.1811447, -106.1534882, 106.1535034
20: -69.4813843, 53.1743393, -69.4813843, 53.1743393, -122.6557159, 122.6557236
21: -84.8355408, 53.7853394, -84.8355408, 53.7853394, -138.6208801, 138.6208801
22: -88.9120178, 52.7000275, -88.9120178, 52.7000275, -141.6120300, 141.6120453
23: -67.5456543, 57.7058563, -67.5456543, 57.7058563, -125.2515106, 125.2515106
24: -88.2593384, 66.0144196, -88.2593384, 66.0144196, -154.2737274, 154.2737274
25: -77.9002457, 60.1505051, -77.9002457, 60.1505051, -138.0507507, 138.0507507
26: -109.3970261, 90.6831818, -109.3970261, 90.6831818, -200.0802002, 200.0802002
27: -88.0051117, 66.4275208, -88.0051117, 66.4275208, -154.4326324, 154.4326324
28: -69.1465912, 62.4998474, -69.1465912, 62.4998474, -131.6464386, 131.6464386
29: -90.2856140, 47.3662109, -90.2856140, 47.3662109, -137.6518097, 137.6518250
30: -89.2205734, 74.5660553, -89.2205734, 74.5660553, -163.7866211, 163.7866211
31: -87.1298294, 64.0869904, -87.1298294, 64.0869904, -151.2168274, 151.2168121
32: -100.1118546, 64.6576691, -100.1118546, 64.6576691, -164.7695312, 164.7695312
33: -133.0290833, 83.3148499, -133.0290833, 83.3148499, -216.3439331, 216.3439331
34: -117.4673386, 62.7784157, -117.4673386, 62.7784157, -180.2457581, 180.2457581
35: -108.4649048, 71.5093231, -108.4649048, 71.5093231, -179.9742279, 179.9742279
36: -111.1997452, 70.1994781, -111.1997452, 70.1994781, -181.3992310, 181.3992310
37: -152.0449677, 71.0758362, -152.0449677, 71.0758362, -223.1207886, 223.1207886
38: -133.2972717, 84.8709869, -133.2972717, 84.8709869, -218.1682587, 218.1682587
39: -149.1843262, 87.7836761, -149.1843262, 87.7836761, -236.9680023, 236.9679871
40: -115.5243454, 66.5658417, -115.5243454, 66.5658417, -182.0901794, 182.0901794
41: -105.0462570, 75.5291519, -105.0462570, 75.5291519, -180.5753937, 180.5754089
42: -76.4584351, 57.2741852, -76.4584351, 57.2741852, -133.7326202, 133.7326202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1623

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9755110, upper bound: 87.9240491
time: 211.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.9481293, upper bound: 87.9701952
time: 170.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 384.69 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 384.69
Output dim: 19, lower bound: -87.8844043, upper bound: 87.8761115
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 384.69
Output dim: 19, lower bound: -87.8271395, upper bound: 87.8785702
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 384.69
Output dim: 19, lower bound: -87.9286216, upper bound: 87.8463147
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 384.69
Output dim: 19, lower bound: -87.8271395, upper bound: 87.8486294
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 384.69
Output dim: 19, lower bound: -87.9755110, upper bound: 87.9240490
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 384.69
Output dim: 19, lower bound: -87.9481293, upper bound: 87.9701951
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 384.69
Output dim: 19, lower bound: -87.9755110, upper bound: 87.9240491
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 384.69
Output dim: 19, lower bound: -87.9481293, upper bound: 87.9701952

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -131.9459839, 77.5789261, -132.2172852, 77.6679916, -209.6139679, 209.7962036
1: -67.2704391, 55.9212646, -67.3591003, 56.0022354, -123.2726746, 123.2803497
2: -56.9108810, 60.2690010, -57.0594635, 60.3134613, -117.2243423, 117.3284607
3: -73.5942078, 69.9391098, -73.9301605, 70.1300049, -143.7241974, 143.8692627
4: -73.4814301, 69.5512390, -73.6966476, 69.6423492, -143.1237793, 143.2478790
5: -69.5997467, 72.0567932, -69.8910217, 72.1959000, -141.7956390, 141.9478149
6: -100.1580887, 73.2229614, -100.1743393, 73.2365952, -173.3946686, 173.3972778
7: -84.2921753, 67.4968643, -84.4799881, 67.5514908, -151.8436584, 151.9768524
8: -101.4775238, 86.9730377, -101.7006760, 87.0574265, -188.5349426, 188.6737061
9: -71.7406082, 72.2162094, -71.9281845, 72.4481354, -144.1887207, 144.1443787
10: -95.8353500, 87.5787048, -96.0882263, 87.8306732, -183.6660156, 183.6669312
11: -90.1115875, 58.5713043, -90.3310928, 58.7218475, -148.8334351, 148.9024048
12: -104.6924057, 89.3655396, -104.6136932, 89.6855087, -194.3778992, 193.9792328
13: -111.9227066, 98.7933426, -111.9288483, 98.9859314, -210.9086151, 210.7221985
14: -160.2234650, 76.2868118, -160.4279785, 76.3943176, -236.6177521, 236.7147827
15: -79.9538879, 66.4699249, -80.1614227, 66.6823730, -146.6362610, 146.6313171
16: -99.2759323, 71.4611359, -99.4319000, 71.5697556, -170.8456879, 170.8930359
17: -157.5835571, 74.1758728, -157.8215485, 74.3085632, -231.8921204, 231.9974213
18: -99.4596786, 87.9564362, -99.6506119, 88.0318604, -187.4915161, 187.6070557
19: -64.7192993, 41.0659866, -64.9321899, 41.0789337, -105.7982330, 105.9981689
20: -69.2722015, 53.0268021, -69.4292984, 53.0856285, -122.3578339, 122.4560928
21: -84.5784760, 53.6529694, -84.7784653, 53.6853180, -138.2637939, 138.4314270
22: -88.7128983, 52.5683975, -88.8401108, 52.6207428, -141.3336487, 141.4085083
23: -67.2439423, 57.4443512, -67.5023193, 57.5724258, -124.8163452, 124.9466705
24: -87.9485550, 65.7078552, -88.1886139, 65.7757950, -153.7243347, 153.8964691
25: -77.6295929, 59.9087029, -77.8516235, 59.9841156, -137.6137085, 137.7603302
26: -109.1489792, 90.3181610, -109.3118668, 90.5685272, -199.7174988, 199.6300049
27: -87.7315598, 66.0982971, -87.9337006, 66.2154694, -153.9470215, 154.0319977
28: -68.8921661, 62.2727051, -69.1045761, 62.3605690, -131.2527313, 131.3772888
29: -90.0716705, 47.1956329, -90.2033920, 47.2974968, -137.3691711, 137.3990173
30: -88.9569092, 74.2564697, -89.1549530, 74.3686218, -163.3255310, 163.4114075
31: -86.8004990, 63.9602585, -87.0704498, 63.9421501, -150.7426453, 151.0307007
32: -99.9473495, 64.5805130, -99.8385544, 64.6052322, -164.5525818, 164.4190674
33: -132.7038879, 83.1970673, -132.8795471, 83.2757339, -215.9796143, 216.0766144
34: -117.2749023, 62.6764526, -117.3870468, 62.7456551, -180.0205536, 180.0635071
35: -108.2404099, 71.3523788, -108.3663101, 71.4724121, -179.7128296, 179.7186890
36: -111.0075378, 70.0144043, -111.0504532, 70.1598434, -181.1673889, 181.0648499
37: -151.6844788, 70.7810593, -151.8879089, 71.0106354, -222.6951141, 222.6689758
38: -133.0330200, 84.6217880, -133.1656189, 84.8050537, -217.8380737, 217.7873840
39: -148.8943787, 87.6399994, -148.9807434, 87.7481537, -236.6425323, 236.6207428
40: -115.2737427, 66.5224304, -115.3705673, 66.5324326, -181.8061676, 181.8929749
41: -104.8483505, 75.3979874, -104.8451385, 75.4836578, -180.3320007, 180.2431030
42: -76.2913055, 57.2081871, -76.2540131, 57.2298317, -133.5211182, 133.4622040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=716, inp2_unstable=717, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.8247764, upper bound: 87.8761115
time: 262.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.8247764, upper bound: 87.8761115
time: 192.10 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -132.1517181, 77.9387512, -132.2028656, 77.6643829, -209.8161011, 210.1416168
1: -67.3562775, 56.0904427, -67.3538666, 55.9991226, -123.3554001, 123.4443054
2: -57.0111237, 60.5787926, -57.0527344, 60.3100014, -117.3211212, 117.6315079
3: -73.7852631, 70.3608093, -73.9214859, 70.1242905, -143.9095459, 144.2822876
4: -73.5839691, 69.8657227, -73.6888504, 69.6392059, -143.2231750, 143.5545654
5: -69.7448883, 72.4553909, -69.8811646, 72.1910553, -141.9359131, 142.3365479
6: -100.4592743, 73.3351288, -100.1696320, 73.2256622, -173.6849365, 173.5047607
7: -84.4542923, 67.7119293, -84.4721146, 67.5460892, -152.0003815, 152.1840515
8: -101.6080017, 87.1908112, -101.6927795, 87.0525970, -188.6605988, 188.8835907
9: -71.9118347, 72.6004257, -71.9195251, 72.4435425, -144.3553772, 144.5199585
10: -96.0527725, 87.9464035, -96.0819016, 87.8242569, -183.8770142, 184.0282898
11: -90.7170181, 58.6577492, -90.3255386, 58.7038841, -149.4208984, 148.9832764
12: -105.2954025, 89.6519165, -104.6072845, 89.6705017, -194.9658813, 194.2592010
13: -111.9906311, 99.1133423, -111.9111938, 98.9789505, -210.9695740, 211.0245361
14: -160.8999023, 76.5167465, -160.4183044, 76.3823853, -237.2822571, 236.9350586
15: -80.0759888, 66.7612610, -80.1442490, 66.6780930, -146.7540588, 146.9054871
16: -99.5355911, 71.6940613, -99.4249802, 71.5640411, -171.0996399, 171.1190491
17: -158.4565430, 74.4760284, -157.8133545, 74.2939758, -232.7505188, 232.2893677
18: -99.9923248, 88.1321106, -99.6419907, 88.0085602, -188.0008545, 187.7740784
19: -65.0637741, 41.1738548, -64.9266663, 41.0771866, -106.1409607, 106.1005173
20: -69.6010437, 53.1021957, -69.4244690, 53.0769424, -122.6779861, 122.5266647
21: -85.0651474, 53.7147064, -84.7725601, 53.6765785, -138.7417297, 138.4872742
22: -89.0819855, 52.6819305, -88.8324432, 52.6169624, -141.6989288, 141.5143738
23: -67.5367279, 57.5497322, -67.4975891, 57.5692253, -125.1059418, 125.0473175
24: -88.2655182, 65.7780304, -88.1804886, 65.7690430, -154.0345612, 153.9585266
25: -77.8187485, 60.0109482, -77.8441925, 59.9809685, -137.7997131, 137.8551331
26: -109.8482285, 90.5480118, -109.3027725, 90.5559998, -200.4042053, 199.8507690
27: -88.2158890, 66.1372833, -87.9262772, 66.1975937, -154.4134827, 154.0635681
28: -69.2207336, 62.3485603, -69.0998840, 62.3571548, -131.5778809, 131.4484406
29: -90.5857544, 47.2861938, -90.1965332, 47.2926712, -137.8784180, 137.4827271
30: -89.3457413, 74.3645325, -89.1491928, 74.3519058, -163.6976471, 163.5137177
31: -87.1023026, 64.0766144, -87.0632782, 63.9398232, -151.0421143, 151.1398926
32: -100.1544724, 64.7026443, -99.8333359, 64.5995941, -164.7540588, 164.5359802
33: -132.9060059, 83.7586212, -132.8677063, 83.2713470, -216.1773529, 216.6263123
34: -117.4413605, 63.0608673, -117.3789520, 62.7418098, -180.1831665, 180.4398193
35: -108.3689575, 71.6312103, -108.3522949, 71.4691925, -179.8381348, 179.9835052
36: -111.3031387, 70.1246033, -111.0453033, 70.1573563, -181.4604797, 181.1698914
37: -152.0554504, 70.8753510, -151.8779602, 71.0012054, -223.0566406, 222.7533112
38: -133.3287506, 84.7904816, -133.1569519, 84.7962723, -218.1250305, 217.9474335
39: -149.0469360, 88.0840149, -148.9688721, 87.7452850, -236.7922211, 237.0528870
40: -115.5020599, 66.6474609, -115.3634033, 66.5292358, -182.0312958, 182.0108643
41: -105.0518112, 75.5530319, -104.8398895, 75.4794617, -180.5312653, 180.3929138
42: -76.4683914, 57.4012680, -76.2496338, 57.2246437, -133.6930237, 133.6508942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=717, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.7703544, upper bound: 87.8645983
time: 209.27 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -87.7703544, upper bound: 87.8609693
time: 740.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 952.28 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 952.28
Output dim: 19, lower bound: -87.8247764, upper bound: 87.8761115
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 952.28
Output dim: 19, lower bound: -87.8247764, upper bound: 87.8761115
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 952.28
Output dim: 19, lower bound: -87.7703544, upper bound: 87.8645983
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 952.28
Output dim: 19, lower bound: -87.7703544, upper bound: 87.8609693
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 952.28
Output dim: 19, lower bound: -87.9286216, upper bound: 87.8463147
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 952.28
Output dim: 19, lower bound: -87.8271395, upper bound: 87.8486294
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 952.28
Output dim: 19, lower bound: -87.9755110, upper bound: 87.9240490
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 952.28
Output dim: 19, lower bound: -87.9481293, upper bound: 87.9701951
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 952.28
Output dim: 19, lower bound: -87.9755110, upper bound: 87.9240491
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 952.28
Output dim: 19, lower bound: -87.9481293, upper bound: 87.9701952
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=106.16907501220703
rel_dist={19: [-88.03682671653708, 88.03682671776684]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 627

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.7133595, upper bound: 86.6919605
time: 141.02 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.7133595, upper bound: 86.7133594
time: 168.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 309.22 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 309.22
Output dim: 19, lower bound: -86.7133595, upper bound: 86.6919605
IS_A2, status: Status.UNKNOWN, split count: 1, time: 309.22
Output dim: 19, lower bound: -86.7133595, upper bound: 86.7133594

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -132.0718079, 77.6194534, -132.2588501, 77.7604065, -209.8321991, 209.8782959
1: -67.3108826, 55.9463425, -67.3765259, 56.0546532, -123.3655319, 123.3228683
2: -56.9860611, 60.2972565, -57.0719719, 60.3733368, -117.3593750, 117.3692169
3: -73.7489014, 69.9801331, -73.9401703, 70.1626129, -143.9114990, 143.9202881
4: -73.5577011, 69.5926971, -73.7083130, 69.7531433, -143.3108521, 143.3010101
5: -69.7194443, 72.0991135, -69.8927612, 72.2667389, -141.9861755, 141.9918823
6: -100.2175293, 73.2617950, -100.3300018, 73.2608032, -173.4783325, 173.5917969
7: -84.3854980, 67.5366058, -84.4969635, 67.6308670, -152.0163574, 152.0335693
8: -101.5638351, 87.0094604, -101.7066650, 87.1564713, -188.7203064, 188.7161255
9: -71.8535538, 72.2626801, -72.0263290, 72.4759216, -144.3294678, 144.2890015
10: -95.8962021, 87.6484070, -96.1298828, 87.8824615, -183.7786560, 183.7782898
11: -90.1747513, 58.6748085, -90.3834610, 58.7866135, -148.9613495, 149.0582581
12: -104.7622833, 89.6147766, -104.8402710, 89.7242432, -194.4864960, 194.4550171
13: -111.9980164, 98.8696289, -112.1504517, 99.0232544, -211.0212708, 211.0200806
14: -160.3263550, 76.5023117, -160.4877014, 76.5974579, -236.9237518, 236.9899902
15: -80.0462723, 66.5190582, -80.1922302, 66.7220001, -146.7682495, 146.7112885
16: -99.3466339, 71.4986572, -99.4814224, 71.6344147, -170.9810333, 170.9800720
17: -157.6744843, 74.4459991, -157.8654785, 74.5541840, -232.2286377, 232.3114624
18: -99.5259781, 88.1391449, -99.6984406, 88.2745056, -187.8004761, 187.8375854
19: -64.7676392, 41.0882874, -64.9537964, 41.1612587, -105.9288940, 106.0420837
20: -69.3254242, 53.0810738, -69.4657440, 53.1550217, -122.4804459, 122.5468140
21: -84.6373825, 53.6940536, -84.8126221, 53.7808113, -138.4181824, 138.5066833
22: -88.7699509, 52.6141624, -88.8939133, 52.6858673, -141.4558105, 141.5080566
23: -67.2905960, 57.4670525, -67.5277863, 57.6436310, -124.9342270, 124.9948425
24: -88.0033340, 65.7506714, -88.2455139, 65.9445953, -153.9479218, 153.9961853
25: -77.6770020, 59.9489326, -77.8850555, 60.1005745, -137.7775726, 137.8339844
26: -109.2186737, 90.5307465, -109.3748016, 90.6558838, -199.8745422, 199.9055481
27: -87.8052216, 66.1688385, -87.9933701, 66.3602600, -154.1654663, 154.1622009
28: -68.9475250, 62.3041115, -69.1334381, 62.4520912, -131.3996124, 131.4375458
29: -90.1294479, 47.2531052, -90.2665176, 47.3393059, -137.4687500, 137.5196228
30: -89.0129395, 74.3575745, -89.2071838, 74.5175171, -163.5304565, 163.5647583
31: -86.8671341, 63.9941406, -87.1053162, 64.0756531, -150.9427795, 151.0994415
32: -100.0142975, 64.6272888, -100.1174164, 64.6429443, -164.6572266, 164.7447052
33: -132.8480835, 83.2524719, -133.0103149, 83.2995071, -216.1475830, 216.2627869
34: -117.3650360, 62.7112541, -117.4582520, 62.7601318, -180.1251678, 180.1695099
35: -108.3294449, 71.3791428, -108.4521942, 71.4765625, -179.8059998, 179.8313293
36: -111.0593414, 70.0432739, -111.1894608, 70.1596680, -181.2190094, 181.2327271
37: -151.7686462, 70.8333969, -152.0216675, 71.0150757, -222.7837219, 222.8550720
38: -133.1062622, 84.6801224, -133.2782440, 84.8237228, -217.9299774, 217.9583435
39: -148.9757080, 87.6772079, -149.1589355, 87.7588730, -236.7345886, 236.8361511
40: -115.3540039, 66.5429535, -115.5123138, 66.5469894, -181.9010010, 182.0552673
41: -104.9207230, 75.4301758, -105.0346298, 75.5079269, -180.4286346, 180.4648132
42: -76.3620453, 57.2464676, -76.4655228, 57.2622070, -133.6242523, 133.7119904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=526, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=717, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6204718, upper bound: 86.6561536
time: 206.17 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6204718, upper bound: 86.6474513
time: 156.91 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -132.3091125, 77.7714996, -132.3177643, 77.7789688, -210.0880585, 210.0892639
1: -67.3951111, 56.0654373, -67.3994751, 56.0707626, -123.4658737, 123.4648972
2: -57.0985336, 60.3796921, -57.1030312, 60.3842201, -117.4827499, 117.4827194
3: -74.0066071, 70.1723938, -74.0153961, 70.1767960, -144.1833801, 144.1877899
4: -73.7588806, 69.7627411, -73.7650146, 69.7691498, -143.5280151, 143.5277557
5: -69.9503098, 72.2761917, -69.9580231, 72.2807465, -142.2310333, 142.2342072
6: -100.3205719, 73.2786713, -100.3517532, 73.2829056, -173.6034698, 173.6304321
7: -84.5334396, 67.6393738, -84.5399246, 67.6439896, -152.1774292, 152.1792908
8: -101.7537994, 87.1708069, -101.7613983, 87.1758423, -188.9296265, 188.9321899
9: -72.0829010, 72.4865265, -72.0906067, 72.4920044, -144.5748901, 144.5771332
10: -96.2020493, 87.8952942, -96.2117233, 87.9024048, -184.1044312, 184.1070099
11: -90.4096069, 58.8208885, -90.4172974, 58.8272552, -149.2368622, 149.2381897
12: -104.8503036, 89.7528305, -104.8588409, 89.7587967, -194.6091003, 194.6116638
13: -112.1929169, 99.0453033, -112.2050629, 99.0518036, -211.2447205, 211.2503662
14: -160.5264740, 76.6091156, -160.5356445, 76.6245422, -237.1510162, 237.1447449
15: -80.2358475, 66.7335510, -80.2425156, 66.7398987, -146.9757080, 146.9760590
16: -99.5118332, 71.6465149, -99.5196075, 71.6526642, -171.1644897, 171.1661224
17: -157.9189301, 74.5764618, -157.9280396, 74.5824661, -232.5013733, 232.5045013
18: -99.7268982, 88.3162308, -99.7324982, 88.3269730, -188.0538635, 188.0487366
19: -64.9723663, 41.1811447, -64.9771118, 41.1896019, -106.1619339, 106.1582336
20: -69.4813843, 53.1743393, -69.4869232, 53.1785812, -122.6599655, 122.6612549
21: -84.8355408, 53.7853394, -84.8414230, 53.8078880, -138.6434326, 138.6267700
22: -88.9120178, 52.7000275, -88.9178543, 52.7096710, -141.6216888, 141.6178894
23: -67.5456543, 57.7058563, -67.5510178, 57.7146416, -125.2602921, 125.2568741
24: -88.2593384, 66.0144196, -88.2655487, 66.0237885, -154.2831116, 154.2799530
25: -77.9002457, 60.1505051, -77.9061584, 60.1584854, -138.0587311, 138.0566711
26: -109.3970261, 90.6831818, -109.4047089, 90.6994629, -200.0964508, 200.0878906
27: -88.0051117, 66.4275208, -88.0118103, 66.4363098, -154.4414062, 154.4393311
28: -69.1465912, 62.4998474, -69.1508484, 62.5061111, -131.6527100, 131.6506958
29: -90.2856140, 47.3662109, -90.2922440, 47.3711319, -137.6567383, 137.6584473
30: -89.2205734, 74.5660553, -89.2284698, 74.5724106, -163.7929840, 163.7945251
31: -87.1298294, 64.0869904, -87.1353760, 64.1042175, -151.2340393, 151.2223511
32: -100.1118546, 64.6576691, -100.1392365, 64.6623688, -164.7742157, 164.7969055
33: -133.0290833, 83.3148499, -133.0384827, 83.3195953, -216.3486633, 216.3533325
34: -117.4673386, 62.7784157, -117.4789734, 62.7826004, -180.2499390, 180.2573853
35: -108.4649048, 71.5093231, -108.4716492, 71.5150757, -179.9799805, 179.9809570
36: -111.1997452, 70.1994781, -111.2071915, 70.2040787, -181.4038239, 181.4066620
37: -152.0449677, 71.0758362, -152.0560608, 71.0840302, -223.1289978, 223.1318970
38: -133.2972717, 84.8709869, -133.3053436, 84.8780746, -218.1753540, 218.1763306
39: -149.1843262, 87.7836761, -149.1932678, 87.7897797, -236.9741058, 236.9769287
40: -115.5243454, 66.5658417, -115.5527039, 66.5701599, -182.0945129, 182.1185455
41: -105.0462570, 75.5291519, -105.0552750, 75.5335846, -180.5798340, 180.5844116
42: -76.4584351, 57.2741852, -76.4859467, 57.2779732, -133.7364044, 133.7601166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=526, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6204718, upper bound: 86.6776111
time: 231.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.6204718, upper bound: 86.6686821
time: 151.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 385.49 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 385.49
Output dim: 19, lower bound: -86.6204718, upper bound: 86.6561536
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 385.49
Output dim: 19, lower bound: -86.6204718, upper bound: 86.6474513
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 385.49
Output dim: 19, lower bound: -86.6204718, upper bound: 86.6776111
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 385.49
Output dim: 19, lower bound: -86.6204718, upper bound: 86.6686821

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -132.0590210, 77.5995712, -132.1978149, 77.6615753, -209.7205963, 209.7973633
1: -67.3056183, 55.9348373, -67.3515167, 55.9967232, -123.3023376, 123.2863541
2: -56.9813499, 60.2833939, -57.0492592, 60.3097267, -117.2910767, 117.3326492
3: -73.7414703, 69.9723358, -73.9053345, 70.1252289, -143.8666992, 143.8776703
4: -73.5508270, 69.5696106, -73.6779327, 69.6368179, -143.1876221, 143.2475433
5: -69.7147522, 72.0832443, -69.8694458, 72.1911392, -141.9058838, 141.9526825
6: -100.1852875, 73.2552948, -100.1657104, 73.2291870, -173.4144745, 173.4210052
7: -84.3792114, 67.5191803, -84.4659119, 67.5469818, -151.9261932, 151.9850922
8: -101.5588913, 86.9879074, -101.6826630, 87.0508881, -188.6097565, 188.6705627
9: -71.8280640, 72.2558289, -71.9070053, 72.4426498, -144.2706909, 144.1628418
10: -95.8819885, 87.6362000, -96.0613022, 87.8238678, -183.7058411, 183.6975098
11: -90.1617661, 58.6580429, -90.3198318, 58.7083168, -148.8700867, 148.9778748
12: -104.7155914, 89.6047668, -104.6072922, 89.6740875, -194.3896637, 194.2120667
13: -111.9474182, 98.8601532, -111.9105835, 98.9763336, -210.9237518, 210.7707214
14: -160.3102722, 76.4579468, -160.4120331, 76.3849106, -236.6951904, 236.8699493
15: -80.0367126, 66.5098190, -80.1447144, 66.6762695, -146.7129822, 146.6545410
16: -99.3338318, 71.4842072, -99.4193268, 71.5635300, -170.8973541, 170.9035339
17: -157.6605072, 74.3942871, -157.8007660, 74.2991409, -231.9596558, 232.1950531
18: -99.5137634, 88.0850372, -99.6392517, 88.0146942, -187.5284271, 187.7242889
19: -64.7614594, 41.0699844, -64.9244232, 41.0692253, -105.8306580, 105.9944077
20: -69.3166351, 53.0653954, -69.4221649, 53.0778198, -122.3944550, 122.4875641
21: -84.6282349, 53.6721802, -84.7688522, 53.6753540, -138.3035889, 138.4410248
22: -88.7571793, 52.5990639, -88.8320236, 52.6126328, -141.3698120, 141.4310913
23: -67.2837219, 57.4473610, -67.4945374, 57.5488510, -124.8325653, 124.9418945
24: -87.9903717, 65.7106476, -88.1818161, 65.7495193, -153.7398682, 153.8924561
25: -77.6684723, 59.9210396, -77.8444824, 59.9649200, -137.6333771, 137.7655182
26: -109.2037201, 90.5092010, -109.3017654, 90.5536118, -199.7573242, 199.8109589
27: -87.7918472, 66.1335983, -87.9273834, 66.1902695, -153.9821167, 154.0609741
28: -68.9403076, 62.2822571, -69.0987091, 62.3427620, -131.2830658, 131.3809662
29: -90.1146011, 47.2412376, -90.1947250, 47.2869873, -137.4015808, 137.4359436
30: -89.0007019, 74.3216782, -89.1476669, 74.3504410, -163.3511353, 163.4693451
31: -86.8578339, 63.9641685, -87.0604553, 63.9319878, -150.7898254, 151.0245972
32: -99.9558563, 64.6180573, -99.8301392, 64.5987396, -164.5545959, 164.4481964
33: -132.8181763, 83.2461090, -132.8699646, 83.2689667, -216.0871277, 216.1160736
34: -117.3488922, 62.7066040, -117.3797150, 62.7382202, -180.0871124, 180.0863190
35: -108.3097229, 71.3756714, -108.3596878, 71.4597015, -179.7694244, 179.7353516
36: -111.0299606, 70.0402756, -111.0443649, 70.1452560, -181.1752167, 181.0846405
37: -151.7391357, 70.8277740, -151.8762207, 70.9877472, -222.7268372, 222.7039642
38: -133.0813446, 84.6726151, -133.1564026, 84.7873230, -217.8686676, 217.8290100
39: -148.9355011, 87.6728516, -148.9692688, 87.7378311, -236.6733093, 236.6421204
40: -115.3228989, 66.5383606, -115.3560257, 66.5246811, -181.8475342, 181.8943787
41: -104.8811264, 75.4234161, -104.8379974, 75.4751282, -180.3562469, 180.2614136
42: -76.3184814, 57.2387505, -76.2460175, 57.2245369, -133.5430145, 133.4847717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=716, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5797318, upper bound: 86.5700375
time: 144.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5401639, upper bound: 86.5769910
time: 289.21 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -132.0643005, 77.6015396, -132.3300018, 77.7616425, -209.8259277, 209.9315186
1: -67.3079681, 55.9382439, -67.4180603, 56.0676041, -123.3755722, 123.3563080
2: -56.9836540, 60.2911110, -57.1330605, 60.4024620, -117.3861160, 117.4241714
3: -73.7432709, 69.9747009, -73.9885254, 70.2040863, -143.9473572, 143.9632263
4: -73.5529938, 69.5765533, -73.8002472, 69.7623901, -143.3153839, 143.3768005
5: -69.7169495, 72.0909271, -69.9764862, 72.3075104, -142.0244598, 142.0674133
6: -100.2081070, 73.2589111, -100.3549042, 73.3932648, -173.6013794, 173.6138000
7: -84.3829880, 67.5299225, -84.5902557, 67.6472244, -152.0302124, 152.1201782
8: -101.5614624, 87.0011749, -101.7926636, 87.1810532, -188.7425232, 188.7938385
9: -71.8447342, 72.2589874, -72.0593262, 72.6258698, -144.4705963, 144.3182983
10: -95.8833237, 87.6434784, -96.1628876, 87.9457321, -183.8290558, 183.8063354
11: -90.1707916, 58.6662827, -90.5501404, 58.8134270, -148.9842224, 149.2164001
12: -104.7532196, 89.6112900, -104.8732452, 89.9378433, -194.6910706, 194.4845123
13: -111.9879379, 98.8645172, -112.1923523, 99.3260117, -211.3139496, 211.0568695
14: -160.3186035, 76.4932251, -160.6667633, 76.5944824, -236.9130554, 237.1599884
15: -80.0393524, 66.5160217, -80.2327881, 66.7772827, -146.8166351, 146.7488098
16: -99.3418274, 71.4914627, -99.6476212, 71.6648788, -171.0066986, 171.1390839
17: -157.6674194, 74.4327850, -158.1526947, 74.5613403, -232.2287598, 232.5854645
18: -99.5178680, 88.1206741, -100.0083847, 88.2833557, -187.8012085, 188.1290588
19: -64.7640533, 41.0849762, -65.0909271, 41.1784172, -105.9424744, 106.1758881
20: -69.3221970, 53.0771179, -69.5404587, 53.1666298, -122.4888229, 122.6175690
21: -84.6335831, 53.6877556, -84.9499283, 53.7904625, -138.4240417, 138.6376801
22: -88.7648087, 52.6064339, -89.0215912, 52.6972809, -141.4620667, 141.6280212
23: -67.2878418, 57.4609299, -67.6487274, 57.6663589, -124.9542007, 125.1096573
24: -87.9977188, 65.7439117, -88.4662476, 65.9727173, -153.9704285, 154.2101593
25: -77.6729889, 59.9425621, -77.9799118, 60.1163788, -137.7893372, 137.9224701
26: -109.2127304, 90.5158844, -109.5733719, 90.6602249, -199.8729553, 200.0892639
27: -87.7995758, 66.1624451, -88.2076187, 66.3822708, -154.1818542, 154.3700562
28: -68.9446869, 62.2990952, -69.2607269, 62.4733620, -131.4180450, 131.5598145
29: -90.1235886, 47.2474937, -90.3640213, 47.3567123, -137.4803009, 137.6115112
30: -89.0089951, 74.3478699, -89.4040604, 74.5436096, -163.5526123, 163.7519226
31: -86.8617401, 63.9889412, -87.2687912, 64.1004028, -150.9621429, 151.2577209
32: -100.0026855, 64.6233368, -100.1391373, 64.8635864, -164.8662720, 164.7624664
33: -132.8378296, 83.2497635, -133.0489349, 83.4624786, -216.3002777, 216.2987061
34: -117.3595505, 62.7071762, -117.5044250, 62.7946014, -180.1541443, 180.2116089
35: -108.3217392, 71.3774109, -108.4849854, 71.5355377, -179.8572693, 179.8623962
36: -111.0471191, 70.0420456, -111.2048798, 70.2837372, -181.3308563, 181.2469177
37: -151.7545166, 70.8313217, -152.0614471, 71.0965576, -222.8510590, 222.8927612
38: -133.0936890, 84.6776352, -133.3092194, 84.9451294, -218.0388184, 217.9868469
39: -148.9569397, 87.6758499, -149.1706848, 87.9563828, -236.9133301, 236.8465271
40: -115.3384933, 66.5409927, -115.5371704, 66.6722946, -182.0107574, 182.0781555
41: -104.9128876, 75.4278870, -105.0600586, 75.6435089, -180.5563965, 180.4879456
42: -76.3531647, 57.2423019, -76.4811859, 57.5024223, -133.8555908, 133.7234802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5797318, upper bound: 86.5581122
time: 271.08 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5401639, upper bound: 86.5623103
time: 175.21 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -132.2963409, 77.7516251, -132.2567291, 77.6801300, -209.9764709, 210.0083618
1: -67.3898392, 56.0539665, -67.3744507, 56.0128250, -123.4026642, 123.4284058
2: -57.0938225, 60.3658333, -57.0803223, 60.3206100, -117.4144287, 117.4461517
3: -73.9991684, 70.1645813, -73.9805603, 70.1394043, -144.1385651, 144.1451416
4: -73.7519989, 69.7396393, -73.7346344, 69.6528625, -143.4048462, 143.4742737
5: -69.9456100, 72.2603226, -69.9346924, 72.2051315, -142.1507416, 142.1950073
6: -100.2883606, 73.2722015, -100.1874847, 73.2512817, -173.5396423, 173.4596710
7: -84.5271301, 67.6219177, -84.5088654, 67.5601196, -152.0872498, 152.1307831
8: -101.7488098, 87.1492691, -101.7373657, 87.0702744, -188.8190765, 188.8866272
9: -72.0574112, 72.4796906, -71.9712296, 72.4587402, -144.5161438, 144.4509277
10: -96.1878433, 87.8831177, -96.1431656, 87.8438263, -184.0316772, 184.0262756
11: -90.3966293, 58.8041267, -90.3536682, 58.7489510, -149.1455841, 149.1577911
12: -104.8036652, 89.7427521, -104.6258621, 89.7086029, -194.5122681, 194.3685913
13: -112.1423645, 99.0358124, -111.9652634, 99.0048828, -211.1472015, 211.0010681
14: -160.5103607, 76.5647049, -160.4599762, 76.4119797, -236.9223328, 237.0246887
15: -80.2262726, 66.7243042, -80.1950226, 66.6941681, -146.9204407, 146.9193268
16: -99.4989777, 71.6320648, -99.4575653, 71.5817719, -171.0807495, 171.0896149
17: -157.9049835, 74.5247498, -157.8634033, 74.3274231, -232.2323608, 232.3881531
18: -99.7147064, 88.2621307, -99.6733322, 88.0673141, -187.7820129, 187.9354553
19: -64.9661942, 41.1628532, -64.9477386, 41.0975761, -106.0637665, 106.1105957
20: -69.4725952, 53.1586494, -69.4433670, 53.1013870, -122.5739822, 122.6020203
21: -84.8263855, 53.7634735, -84.7976837, 53.7024422, -138.5288086, 138.5611572
22: -88.8992310, 52.6849251, -88.8559647, 52.6364326, -141.5356445, 141.5408936
23: -67.5387726, 57.6861687, -67.5177765, 57.6198997, -125.1586761, 125.2039337
24: -88.2463684, 65.9744263, -88.2018356, 65.8287201, -154.0750732, 154.1762543
25: -77.8916931, 60.1226158, -77.8656387, 60.0228577, -137.9145508, 137.9882507
26: -109.3820419, 90.6616516, -109.3316803, 90.5971756, -199.9792175, 199.9933319
27: -87.9917679, 66.3922882, -87.9458389, 66.2663345, -154.2581024, 154.3381348
28: -69.1394196, 62.4780006, -69.1161346, 62.3967972, -131.5362244, 131.5941315
29: -90.2707520, 47.3543243, -90.2204666, 47.3188210, -137.5895691, 137.5747681
30: -89.2083664, 74.5301590, -89.1689682, 74.4053726, -163.6137390, 163.6991272
31: -87.1205673, 64.0569992, -87.0905228, 63.9605598, -151.0811310, 151.1475220
32: -100.0534363, 64.6484375, -99.8519592, 64.6181793, -164.6716156, 164.5003967
33: -132.9991608, 83.3084564, -132.8981018, 83.2890854, -216.2882385, 216.2065582
34: -117.4511719, 62.7737350, -117.4004669, 62.7606888, -180.2118225, 180.1741943
35: -108.4452057, 71.5058594, -108.3791580, 71.4982300, -179.9434052, 179.8850098
36: -111.1703568, 70.1964264, -111.0620575, 70.1897049, -181.3600616, 181.2584839
37: -152.0154419, 71.0702667, -151.9106445, 71.0567017, -223.0721130, 222.9809113
38: -133.2723236, 84.8634491, -133.1834717, 84.8416748, -218.1139832, 218.0469208
39: -149.1441040, 87.7792892, -149.0036011, 87.7687225, -236.9128113, 236.7828674
40: -115.4932327, 66.5612106, -115.3964539, 66.5478821, -182.0411072, 181.9576721
41: -105.0066452, 75.5223999, -104.8585892, 75.5007935, -180.5074463, 180.3809814
42: -76.4148560, 57.2664185, -76.2664413, 57.2403336, -133.6551819, 133.5328674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=718, inp2_unstable=718, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5797318, upper bound: 86.5913615
time: 194.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5401639, upper bound: 86.5983668
time: 175.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -132.3016357, 77.7536087, -132.3889313, 77.7802277, -210.0818634, 210.1425476
1: -67.3921967, 56.0573578, -67.4409943, 56.0837173, -123.4758911, 123.4983444
2: -57.0961342, 60.3735237, -57.1641273, 60.4133568, -117.5094910, 117.5376511
3: -74.0010071, 70.1669769, -74.0637360, 70.2182770, -144.2192688, 144.2307129
4: -73.7541504, 69.7465820, -73.8569336, 69.7784729, -143.5326233, 143.6035156
5: -69.9478226, 72.2680511, -70.0417252, 72.3214874, -142.2693176, 142.3097839
6: -100.3111420, 73.2758102, -100.3766861, 73.4153290, -173.7264709, 173.6524963
7: -84.5309143, 67.6326828, -84.6332550, 67.6603088, -152.1912231, 152.2659302
8: -101.7513962, 87.1625214, -101.8474045, 87.2004395, -188.9518127, 189.0099182
9: -72.0740814, 72.4828339, -72.1235809, 72.6419220, -144.7160034, 144.6064148
10: -96.1891556, 87.8903351, -96.2447433, 87.9656830, -184.1548462, 184.1350708
11: -90.4056473, 58.8123665, -90.5839996, 58.8540573, -149.2597046, 149.3963623
12: -104.8412628, 89.7493362, -104.8918762, 89.9723587, -194.8136292, 194.6412048
13: -112.1828537, 99.0401993, -112.2470245, 99.3545914, -211.5374451, 211.2872314
14: -160.5187073, 76.6000366, -160.7146606, 76.6215820, -237.1402740, 237.3146667
15: -80.2289200, 66.7304993, -80.2831116, 66.7951584, -147.0240479, 147.0135956
16: -99.5069885, 71.6393433, -99.6858139, 71.6831512, -171.1901245, 171.3251343
17: -157.9118652, 74.5632477, -158.2153015, 74.5896301, -232.5014954, 232.7785492
18: -99.7188034, 88.2977600, -100.0424500, 88.3358765, -188.0546875, 188.3401947
19: -64.9687958, 41.1778412, -65.1142502, 41.2067337, -106.1755295, 106.2920914
20: -69.4781494, 53.1703835, -69.5616150, 53.1901894, -122.6683121, 122.7319870
21: -84.8317184, 53.7790604, -84.9787598, 53.8175468, -138.6492615, 138.7578125
22: -88.9068680, 52.6923027, -89.0455704, 52.7210846, -141.6279297, 141.7378693
23: -67.5428925, 57.6997452, -67.6719742, 57.7374115, -125.2803040, 125.3717041
24: -88.2537308, 66.0076752, -88.4862595, 66.0519104, -154.3056335, 154.4939270
25: -77.8962173, 60.1441536, -78.0010071, 60.1743279, -138.0705414, 138.1451569
26: -109.3910675, 90.6683426, -109.6032944, 90.7038269, -200.0948792, 200.2716217
27: -87.9994431, 66.4211349, -88.2260513, 66.4582977, -154.4577332, 154.6471863
28: -69.1437836, 62.4948387, -69.2781372, 62.5273933, -131.6711731, 131.7729797
29: -90.2797241, 47.3605728, -90.3897705, 47.3885307, -137.6682587, 137.7503357
30: -89.2166138, 74.5563660, -89.4253845, 74.5985107, -163.8151245, 163.9817505
31: -87.1244659, 64.0817642, -87.2988815, 64.1289520, -151.2534180, 151.3806458
32: -100.1002426, 64.6536865, -100.1609726, 64.8829956, -164.9832306, 164.8146515
33: -133.0187988, 83.3121262, -133.0771027, 83.4825668, -216.5013733, 216.3892212
34: -117.4618683, 62.7743073, -117.5251541, 62.8170738, -180.2789459, 180.2994690
35: -108.4572144, 71.5075684, -108.5044632, 71.5740662, -180.0312805, 180.0120239
36: -111.1875153, 70.1981812, -111.2226028, 70.3281479, -181.5156555, 181.4207764
37: -152.0308228, 71.0738220, -152.0958405, 71.1655426, -223.1963654, 223.1696625
38: -133.2846832, 84.8684921, -133.3363342, 84.9994659, -218.2841492, 218.2048340
39: -149.1655884, 87.7822647, -149.2050171, 87.9873352, -237.1529083, 236.9872437
40: -115.5088272, 66.5639038, -115.5775757, 66.6954956, -182.2043152, 182.1414795
41: -105.0384140, 75.5268326, -105.0806961, 75.6691589, -180.7075653, 180.6075287
42: -76.4495697, 57.2700195, -76.5016327, 57.5182343, -133.9678040, 133.7716522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=718, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5797318, upper bound: 86.5793943
time: 229.83 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5401639, upper bound: 86.5836631
time: 151.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 383.60 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 383.60
Output dim: 19, lower bound: -86.5797318, upper bound: 86.5700375
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 383.60
Output dim: 19, lower bound: -86.5401639, upper bound: 86.5769910
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 383.60
Output dim: 19, lower bound: -86.5797318, upper bound: 86.5581122
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 383.60
Output dim: 19, lower bound: -86.5401639, upper bound: 86.5623103
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 383.60
Output dim: 19, lower bound: -86.5797318, upper bound: 86.5913615
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 383.60
Output dim: 19, lower bound: -86.5401639, upper bound: 86.5983668
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 383.60
Output dim: 19, lower bound: -86.5797318, upper bound: 86.5793943
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 383.60
Output dim: 19, lower bound: -86.5401639, upper bound: 86.5836631

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -131.9371338, 77.5644608, -132.1978149, 77.6615753, -209.5987091, 209.7622681
1: -67.2668304, 55.9128036, -67.3515167, 55.9967232, -123.2635498, 123.2643204
2: -56.9076004, 60.2588997, -57.0492592, 60.3097267, -117.2173309, 117.3081589
3: -73.5892639, 69.9335175, -73.9053345, 70.1252289, -143.7144928, 143.8388519
4: -73.4771423, 69.5343170, -73.6779327, 69.6368179, -143.1139526, 143.2122498
5: -69.5963898, 72.0452423, -69.8694458, 72.1911392, -141.7875366, 141.9146729
6: -100.1342239, 73.2183151, -100.1657104, 73.2291870, -173.3634033, 173.3840179
7: -84.2876892, 67.4840240, -84.4659119, 67.5469818, -151.8346710, 151.9499359
8: -101.4740219, 86.9573593, -101.6826630, 87.0508881, -188.5249023, 188.6400146
9: -71.7232361, 72.2113419, -71.9070053, 72.4426498, -144.1658936, 144.1183472
10: -95.8250732, 87.5701218, -96.0613022, 87.8238678, -183.6489410, 183.6314240
11: -90.1023178, 58.5599632, -90.3198318, 58.7083168, -148.8106384, 148.8797913
12: -104.6580963, 89.3583069, -104.6072922, 89.6740875, -194.3321838, 193.9656067
13: -111.8876724, 98.7866592, -111.9105835, 98.9763336, -210.8640137, 210.6972351
14: -160.2124176, 76.2563095, -160.4120331, 76.3849106, -236.5973206, 236.6683197
15: -79.9468994, 66.4632416, -80.1447144, 66.6762695, -146.6231689, 146.6079559
16: -99.2669678, 71.4506531, -99.4193268, 71.5635300, -170.8305054, 170.8699799
17: -157.5740967, 74.1395721, -157.8007660, 74.2991409, -231.8731995, 231.9403381
18: -99.4510040, 87.9180908, -99.6392517, 88.0146942, -187.4656830, 187.5573425
19: -64.7149811, 41.0525970, -64.9244232, 41.0692253, -105.7841949, 105.9770203
20: -69.2658844, 53.0154991, -69.4221649, 53.0778198, -122.3437042, 122.4376602
21: -84.5721664, 53.6375618, -84.7688522, 53.6753540, -138.2475281, 138.4064178
22: -88.7038651, 52.5574226, -88.8320236, 52.6126328, -141.3164978, 141.3894501
23: -67.2390747, 57.4306488, -67.4945374, 57.5488510, -124.7879257, 124.9251709
24: -87.9391174, 65.6793976, -88.1818161, 65.7495193, -153.6886292, 153.8612061
25: -77.6235580, 59.8890114, -77.8444824, 59.9649200, -137.5884705, 137.7334900
26: -109.1382599, 90.3032990, -109.3017654, 90.5536118, -199.6918488, 199.6050568
27: -87.7219925, 66.0738525, -87.9273834, 66.1902695, -153.9122620, 154.0012360
28: -68.8871155, 62.2569771, -69.0987091, 62.3427620, -131.2298584, 131.3556824
29: -90.0611801, 47.1883087, -90.1947250, 47.2869873, -137.3481445, 137.3830261
30: -88.9481812, 74.2317963, -89.1476669, 74.3504410, -163.2986145, 163.3794556
31: -86.7939606, 63.9384270, -87.0604553, 63.9319878, -150.7259369, 150.9988708
32: -99.9053726, 64.5739670, -99.8301392, 64.5987396, -164.5041199, 164.4041138
33: -132.6835938, 83.1926727, -132.8699646, 83.2689667, -215.9525452, 216.0626373
34: -117.2635422, 62.6732063, -117.3797150, 62.7382202, -180.0017700, 180.0529175
35: -108.2269135, 71.3499146, -108.3596878, 71.4597015, -179.6866150, 179.7095947
36: -110.9863968, 70.0123138, -111.0443649, 70.1452560, -181.1316528, 181.0566711
37: -151.6630249, 70.7770920, -151.8762207, 70.9877472, -222.6507721, 222.6532898
38: -133.0151672, 84.6166153, -133.1564026, 84.7873230, -217.8024902, 217.7730103
39: -148.8671875, 87.6370010, -148.9692688, 87.7378311, -236.6049805, 236.6062622
40: -115.2510147, 66.5192032, -115.3560257, 66.5246811, -181.7756653, 181.8752289
41: -104.8194656, 75.3932114, -104.8379974, 75.4751282, -180.2945709, 180.2312012
42: -76.2589340, 57.2026634, -76.2460175, 57.2245369, -133.4834595, 133.4486847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=716, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5334590, upper bound: 86.5700375
time: 170.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5334590, upper bound: 86.5700375
time: 175.02 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -132.1428680, 77.9243164, -132.1776581, 77.6565323, -209.7993774, 210.1019745
1: -67.3526840, 56.0820351, -67.3442535, 55.9923706, -123.3450546, 123.4262848
2: -57.0078354, 60.5687218, -57.0397758, 60.3048515, -117.3126831, 117.6084900
3: -73.7803192, 70.3552628, -73.8930969, 70.1171722, -143.8974609, 144.2483215
4: -73.5796356, 69.8488464, -73.6670074, 69.6323853, -143.2120209, 143.5158539
5: -69.7415161, 72.4439774, -69.8555603, 72.1843719, -141.9258881, 142.2995300
6: -100.4354477, 73.3304901, -100.1590500, 73.2138443, -173.6492920, 173.4895325
7: -84.4498062, 67.6990585, -84.4548111, 67.5394745, -151.9892883, 152.1538696
8: -101.6045151, 87.1751099, -101.6715851, 87.0441284, -188.6486359, 188.8466949
9: -71.8944855, 72.5955734, -71.8949509, 72.4361267, -144.3306122, 144.4905090
10: -96.0425262, 87.9378357, -96.0523911, 87.8148041, -183.8573151, 183.9902344
11: -90.7078323, 58.6464119, -90.3120117, 58.6829262, -149.3907471, 148.9584198
12: -105.2610321, 89.6446228, -104.5982666, 89.6529846, -194.9140167, 194.2428894
13: -111.9555893, 99.1066437, -111.8856735, 98.9664383, -210.9220123, 210.9923096
14: -160.8888550, 76.4862823, -160.3983765, 76.3680496, -237.2568970, 236.8846436
15: -80.0689545, 66.7546387, -80.1204834, 66.6702271, -146.7391815, 146.8751221
16: -99.5266113, 71.6836243, -99.4095917, 71.5554810, -171.0820923, 171.0932159
17: -158.4471436, 74.4397354, -157.7892303, 74.2786255, -232.7257690, 232.2289581
18: -99.9836960, 88.0937805, -99.6270676, 87.9830246, -187.9667206, 187.7208557
19: -65.0594788, 41.1604462, -64.9166641, 41.0667801, -106.1262589, 106.0771027
20: -69.5947495, 53.0908966, -69.4153748, 53.0657234, -122.6604767, 122.5062714
21: -85.0588455, 53.6992874, -84.7605438, 53.6631126, -138.7219543, 138.4598389
22: -89.0729675, 52.6709785, -88.8212738, 52.6072807, -141.6802521, 141.4922485
23: -67.5318756, 57.5360298, -67.4878845, 57.5443497, -125.0762253, 125.0239105
24: -88.2561340, 65.7495728, -88.1704178, 65.7401505, -153.9962769, 153.9199829
25: -77.8127289, 59.9912643, -77.8340607, 59.9604912, -137.7732239, 137.8253174
26: -109.8375244, 90.5332489, -109.2889786, 90.5359497, -200.3734589, 199.8222351
27: -88.2064133, 66.1128387, -87.9169006, 66.1649475, -154.3713684, 154.0297241
28: -69.2157135, 62.3328209, -69.0921326, 62.3379555, -131.5536499, 131.4249573
29: -90.5752411, 47.2788658, -90.1850815, 47.2801971, -137.8554382, 137.4639435
30: -89.3371506, 74.3396912, -89.1395798, 74.3272400, -163.6643677, 163.4792786
31: -87.0957718, 64.0547943, -87.0502930, 63.9287491, -151.0245209, 151.1050873
32: -100.1125031, 64.6961212, -99.8227692, 64.5909119, -164.7034149, 164.5188599
33: -132.8856812, 83.7542419, -132.8533783, 83.2628174, -216.1484985, 216.6076202
34: -117.4299622, 63.0576973, -117.3683701, 62.7328186, -180.1627808, 180.4260559
35: -108.3554535, 71.6287842, -108.3402557, 71.4551544, -179.8106079, 179.9690399
36: -111.2820129, 70.1225128, -111.0370636, 70.1417694, -181.4237823, 181.1595764
37: -152.0340271, 70.8713989, -151.8623657, 70.9744263, -223.0084534, 222.7337494
38: -133.3108826, 84.7852402, -133.1442108, 84.7749023, -218.0857849, 217.9294434
39: -149.0197449, 88.0809631, -148.9525909, 87.7337646, -236.7535095, 237.0335541
40: -115.4793701, 66.6442261, -115.3459320, 66.5202408, -181.9996033, 181.9901581
41: -105.0229492, 75.5482941, -104.8306351, 75.4692535, -180.4922028, 180.3789062
42: -76.4360352, 57.3957176, -76.2398529, 57.2172966, -133.6533356, 133.6355743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.4860840, upper bound: 86.5657821
time: 190.41 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.4860840, upper bound: 86.5657821
time: 222.44 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -131.9419556, 77.5661163, -132.3300018, 77.7616425, -209.7035828, 209.8961182
1: -67.2689514, 55.9161110, -67.4180603, 56.0676041, -123.3365555, 123.3341675
2: -56.9097672, 60.2664490, -57.1330605, 60.4024620, -117.3122253, 117.3995056
3: -73.5906830, 69.9357452, -73.9885254, 70.2040863, -143.7947693, 143.9242706
4: -73.4787445, 69.5410461, -73.8002472, 69.7623901, -143.2411346, 143.3412933
5: -69.5985031, 72.0527344, -69.9764862, 72.3075104, -141.9060059, 142.0292053
6: -100.1568604, 73.2217712, -100.3549042, 73.3932648, -173.5501099, 173.5766602
7: -84.2913208, 67.4946136, -84.5902557, 67.6472244, -151.9385376, 152.0848694
8: -101.4764404, 86.9702911, -101.7926636, 87.1810532, -188.6574707, 188.7629547
9: -71.7387543, 72.2143250, -72.0593262, 72.6258698, -144.3646240, 144.2736511
10: -95.8262024, 87.5770035, -96.1628876, 87.9457321, -183.7719269, 183.7398682
11: -90.1110382, 58.5674095, -90.5501404, 58.8134270, -148.9244385, 149.1175232
12: -104.6953125, 89.3646698, -104.8732452, 89.9378433, -194.6331482, 194.2379150
13: -111.9263153, 98.7907867, -112.1923523, 99.3260117, -211.2523193, 210.9831390
14: -160.2200623, 76.2897949, -160.6667633, 76.5944824, -236.8145447, 236.9565430
15: -79.9494476, 66.4692841, -80.2327881, 66.7772827, -146.7267303, 146.7020569
16: -99.2745667, 71.4577179, -99.6476212, 71.6648788, -170.9394379, 171.1053314
17: -157.5803528, 74.1764984, -158.1526947, 74.5613403, -232.1416931, 232.3291931
18: -99.4547882, 87.9522858, -100.0083847, 88.2833557, -187.7381439, 187.9606628
19: -64.7173843, 41.0674362, -65.0909271, 41.1784172, -105.8957977, 106.1583633
20: -69.2712708, 53.0269356, -69.5404587, 53.1666298, -122.4378967, 122.5673828
21: -84.5771408, 53.6524963, -84.9499283, 53.7904625, -138.3675995, 138.6024170
22: -88.7111206, 52.5645828, -89.0215912, 52.6972809, -141.4083862, 141.5861816
23: -67.2430115, 57.4435272, -67.6487274, 57.6663589, -124.9093704, 125.0922394
24: -87.9463043, 65.7116547, -88.4662476, 65.9727173, -153.9190216, 154.1778870
25: -77.6278229, 59.9098015, -77.9799118, 60.1163788, -137.7442017, 137.8897095
26: -109.1469269, 90.3091507, -109.5733719, 90.6602249, -199.8071442, 199.8825073
27: -87.7294083, 66.1014099, -88.2076187, 66.3822708, -154.1116791, 154.3090210
28: -68.8912964, 62.2733803, -69.2607269, 62.4733620, -131.3646393, 131.5341034
29: -90.0697479, 47.1935081, -90.3640213, 47.3567123, -137.4264526, 137.5575256
30: -88.9561920, 74.2565765, -89.4040604, 74.5436096, -163.4998016, 163.6606445
31: -86.7975616, 63.9627876, -87.2687912, 64.1004028, -150.8979645, 151.2315674
32: -99.9510193, 64.5789795, -100.1391373, 64.8635864, -164.8146057, 164.7181091
33: -132.7018890, 83.1960907, -133.0489349, 83.4624786, -216.1643677, 216.2450256
34: -117.2737427, 62.6736298, -117.5044250, 62.7946014, -180.0683441, 180.1780548
35: -108.2380981, 71.3515625, -108.4849854, 71.5355377, -179.7736359, 179.8365479
36: -111.0029755, 70.0139771, -111.2048798, 70.2837372, -181.2867126, 181.2188568
37: -151.6779785, 70.7805023, -152.0614471, 71.0965576, -222.7745209, 222.8419495
38: -133.0269623, 84.6212921, -133.3092194, 84.9451294, -217.9720612, 217.9305115
39: -148.8867645, 87.6398544, -149.1706848, 87.9563828, -236.8431396, 236.8105469
40: -115.2662277, 66.5217361, -115.5371704, 66.6722946, -181.9385223, 182.0588989
41: -104.8507767, 75.3974915, -105.0600586, 75.6435089, -180.4942780, 180.4575500
42: -76.2934494, 57.2060776, -76.4811859, 57.5024223, -133.7958679, 133.6872559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=716, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5334590, upper bound: 86.5581122
time: 148.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -86.5334590, upper bound: 86.5581122
time: 202.15 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -132.1476898, 77.9259033, -132.3098907, 77.7565460, -209.9042358, 210.2357788
1: -67.3548279, 56.0853195, -67.4107971, 56.0632057, -123.4180298, 123.4961090
2: -57.0099945, 60.5762367, -57.1236076, 60.3975601, -117.4075470, 117.6998291
3: -73.7817383, 70.3574219, -73.9762726, 70.1960754, -143.9778137, 144.3336945
4: -73.5812683, 69.8554535, -73.7893372, 69.7579651, -143.3392181, 143.6447906
5: -69.7436218, 72.4513702, -69.9626160, 72.3007355, -142.0443573, 142.4139709
6: -100.4580765, 73.3339539, -100.3482437, 73.3779678, -173.8360443, 173.6821899
7: -84.4534683, 67.7097549, -84.5792236, 67.6397705, -152.0932312, 152.2889709
8: -101.6069107, 87.1879730, -101.7816162, 87.1742859, -188.7811890, 188.9695892
9: -71.9100189, 72.5985260, -72.0472107, 72.6192932, -144.5293121, 144.6457367
10: -96.0436020, 87.9446869, -96.1539383, 87.9367218, -183.9803162, 184.0986176
11: -90.7164459, 58.6538544, -90.5423355, 58.7880707, -149.5045166, 149.1961975
12: -105.2983246, 89.6510620, -104.8642807, 89.9167557, -195.2150879, 194.5153503
13: -111.9943085, 99.1107025, -112.1675186, 99.3160706, -211.3103638, 211.2782135
14: -160.8965149, 76.5197144, -160.6530762, 76.5776367, -237.4741516, 237.1727905
15: -80.0715256, 66.7606430, -80.2085419, 66.7712784, -146.8427734, 146.9691772
16: -99.5342255, 71.6906662, -99.6378937, 71.6569672, -171.1911621, 171.3285522
17: -158.4533386, 74.4766235, -158.1411438, 74.5408325, -232.9941711, 232.6177521
18: -99.9874649, 88.1278687, -99.9961243, 88.2516937, -188.2391663, 188.1239929
19: -65.0618591, 41.1753006, -65.0831833, 41.1759682, -106.2378235, 106.2584686
20: -69.6001205, 53.1023216, -69.5336609, 53.1545067, -122.7546082, 122.6359711
21: -85.0638046, 53.7142105, -84.9416351, 53.7782059, -138.8420105, 138.6558533
22: -89.0802231, 52.6781082, -89.0108337, 52.6919289, -141.7721558, 141.6889343
23: -67.5358047, 57.5489578, -67.6421051, 57.6618538, -125.1976624, 125.1910629
24: -88.2632523, 65.7818146, -88.4548340, 65.9633560, -154.2266083, 154.2366333
25: -77.8169785, 60.0120354, -77.9694824, 60.1119423, -137.9289246, 137.9815063
26: -109.8462219, 90.5390625, -109.5606003, 90.6425323, -200.4887390, 200.0996704
27: -88.2137833, 66.1403580, -88.1970062, 66.3569489, -154.5707397, 154.3373413
28: -69.2198639, 62.3492317, -69.2541504, 62.4686203, -131.6884766, 131.6033783
29: -90.5838470, 47.2840424, -90.3544159, 47.3499069, -137.9337463, 137.6384583
30: -89.3450012, 74.3647156, -89.3959961, 74.5207520, -163.8657532, 163.7607117
31: -87.0993958, 64.0791473, -87.2586060, 64.0971222, -151.1965179, 151.3377380
32: -100.1581650, 64.7010956, -100.1318588, 64.8557587, -165.0139160, 164.8329468
33: -132.9040527, 83.7576447, -133.0323486, 83.4563293, -216.3603668, 216.7899933
34: -117.4402771, 63.0580444, -117.4930420, 62.7892303, -180.2295074, 180.5510864
35: -108.3666916, 71.6304092, -108.4655075, 71.5309753, -179.8976746, 180.0959167
36: -111.2986221, 70.1242294, -111.1976700, 70.2802353, -181.5788574, 181.3218842
37: -152.0490112, 70.8747864, -152.0475769, 71.0832977, -223.1323090, 222.9223633
38: -133.3226929, 84.7900238, -133.2971954, 84.9327621, -218.2554169, 218.0872192
39: -149.0393677, 88.0838165, -149.1540527, 87.9523468, -236.9916992, 237.2378693
40: -115.4945755, 66.6467590, -115.5270538, 66.6678314, -182.1624146, 182.1737976
41: -105.0542297, 75.5525360, -105.0526657, 75.6376572, -180.6918945, 180.6051941
42: -76.4705200, 57.3991280, -76.4750519, 57.4952126, -133.9657288, 133.8741760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=106.16907501220703
rel_dist={19: [-86.72003761785648, 86.72003761772815]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 627

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9440744, upper bound: 85.9240615
time: 135.20 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.9440744, upper bound: 85.9440743
time: 199.14 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 334.48 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 334.48
Output dim: 19, lower bound: -85.9440744, upper bound: 85.9240615
IS_A2, status: Status.UNKNOWN, split count: 1, time: 334.48
Output dim: 19, lower bound: -85.9440744, upper bound: 85.9440743

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -132.0718079, 77.6194534, -132.2472229, 77.7566299, -209.8284302, 209.8666687
1: -67.3108826, 55.9463425, -67.3720398, 56.0514145, -123.3622971, 123.3183823
2: -56.9860611, 60.2972565, -57.0660858, 60.3711205, -117.3571625, 117.3633347
3: -73.7489014, 69.9801331, -73.9255829, 70.1597443, -143.9086456, 143.9057007
4: -73.5577011, 69.5926971, -73.6971283, 69.7498856, -143.3075867, 143.2898254
5: -69.7194443, 72.0991135, -69.8800507, 72.2639542, -141.9833984, 141.9791565
6: -100.2175293, 73.2617950, -100.3249054, 73.2564087, -173.4739380, 173.5866852
7: -84.3854980, 67.5366058, -84.4888000, 67.6282501, -152.0137482, 152.0254059
8: -101.5638351, 87.0094604, -101.6961517, 87.1526337, -188.7164612, 188.7056122
9: -71.8535538, 72.2626801, -72.0136261, 72.4726715, -144.3262177, 144.2763062
10: -95.8962021, 87.6484070, -96.1137695, 87.8784637, -183.7746582, 183.7621765
11: -90.1747513, 58.6748085, -90.3768005, 58.7785416, -148.9532928, 149.0516052
12: -104.7622833, 89.6147766, -104.8364410, 89.7177505, -194.4800262, 194.4512177
13: -111.9980164, 98.8696289, -112.1398697, 99.0175095, -211.0155182, 211.0094910
14: -160.3263550, 76.5023117, -160.4781799, 76.5919418, -236.9182739, 236.9804840
15: -80.0462723, 66.5190582, -80.1822205, 66.7184448, -146.7647095, 146.7012634
16: -99.3466339, 71.4986572, -99.4739914, 71.6307220, -170.9773560, 170.9726257
17: -157.6744843, 74.4459991, -157.8531189, 74.5485916, -232.2230072, 232.2991028
18: -99.5259781, 88.1391449, -99.6916809, 88.2644424, -187.7904205, 187.8308258
19: -64.7676392, 41.0882874, -64.9492416, 41.1554565, -105.9230957, 106.0375214
20: -69.3254242, 53.0810738, -69.4615326, 53.1503525, -122.4757690, 122.5426025
21: -84.6373825, 53.6940536, -84.8069992, 53.7748451, -138.4122314, 138.5010376
22: -88.7699509, 52.6141624, -88.8891525, 52.6812286, -141.4511719, 141.5033112
23: -67.2905960, 57.4670525, -67.5232086, 57.6295090, -124.9201050, 124.9902649
24: -88.0033340, 65.7506714, -88.2415237, 65.9288025, -153.9321289, 153.9921875
25: -77.6770020, 59.9489326, -77.8808441, 60.0892525, -137.7662506, 137.8297729
26: -109.2186737, 90.5307465, -109.3688049, 90.6469727, -199.8656311, 199.8995514
27: -87.8052216, 66.1688385, -87.9896622, 66.3451157, -154.1503296, 154.1585083
28: -68.9475250, 62.3041115, -69.1299744, 62.4414330, -131.3889618, 131.4340668
29: -90.1294479, 47.2531052, -90.2613983, 47.3332520, -137.4627075, 137.5144958
30: -89.0129395, 74.3575745, -89.2029037, 74.5066223, -163.5195618, 163.5604858
31: -86.8671341, 63.9941406, -87.0993576, 64.0696487, -150.9367676, 151.0935059
32: -100.0142975, 64.6272888, -100.1124039, 64.6391220, -164.6534119, 164.7396851
33: -132.8480835, 83.2524719, -133.0046997, 83.2954636, -216.1435394, 216.2571716
34: -117.3650360, 62.7112541, -117.4539719, 62.7557945, -180.1208344, 180.1652222
35: -108.3294449, 71.3791428, -108.4483566, 71.4689636, -179.7984009, 179.8274994
36: -111.0593414, 70.0432739, -111.1859894, 70.1509171, -181.2102661, 181.2292480
37: -151.7686462, 70.8333969, -152.0147705, 71.0013885, -222.7700195, 222.8481750
38: -133.1062622, 84.6801224, -133.2729492, 84.8131485, -217.9194031, 217.9530640
39: -148.9757080, 87.6772079, -149.1522675, 87.7527313, -236.7284393, 236.8294678
40: -115.3540039, 66.5429535, -115.5036392, 66.5424652, -181.8964691, 182.0466003
41: -104.9207230, 75.4301758, -105.0304337, 75.5027924, -180.4235229, 180.4606018
42: -76.3620453, 57.2464676, -76.4607925, 57.2590675, -133.6211090, 133.7072601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=526, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=717, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1633

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8572715, upper bound: 85.8866068
time: 139.02 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8572715, upper bound: 85.8810289
time: 200.14 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -132.3091125, 77.7714996, -132.3168335, 77.7781601, -210.0872803, 210.0883331
1: -67.3951111, 56.0654373, -67.3989868, 56.0701866, -123.4653015, 123.4644165
2: -57.0985336, 60.3796921, -57.1025505, 60.3837242, -117.4822540, 117.4822311
3: -74.0066071, 70.1723938, -74.0144653, 70.1763000, -144.1829071, 144.1868591
4: -73.7588806, 69.7627411, -73.7643433, 69.7684937, -143.5273743, 143.5270691
5: -69.9503098, 72.2761917, -69.9571838, 72.2802582, -142.2305603, 142.2333679
6: -100.3205719, 73.2786713, -100.3484955, 73.2824554, -173.6030273, 173.6271515
7: -84.5334396, 67.6393738, -84.5392303, 67.6434937, -152.1769104, 152.1785889
8: -101.7537994, 87.1708069, -101.7605743, 87.1753006, -188.9291077, 188.9313660
9: -72.0829010, 72.4865265, -72.0897522, 72.4914246, -144.5743256, 144.5762787
10: -96.2020493, 87.8952942, -96.2107010, 87.9016571, -184.1036987, 184.1059875
11: -90.4096069, 58.8208885, -90.4164963, 58.8265572, -149.2361603, 149.2373810
12: -104.8503036, 89.7528305, -104.8579025, 89.7581558, -194.6084595, 194.6107178
13: -112.1929169, 99.0453033, -112.2037201, 99.0511017, -211.2440186, 211.2490234
14: -160.5264740, 76.6091156, -160.5346375, 76.6228485, -237.1493225, 237.1437378
15: -80.2358475, 66.7335510, -80.2417908, 66.7391968, -146.9750366, 146.9753418
16: -99.5118332, 71.6465149, -99.5187454, 71.6520081, -171.1638336, 171.1652527
17: -157.9189301, 74.5764618, -157.9270935, 74.5818024, -232.5007324, 232.5035553
18: -99.7268982, 88.3162308, -99.7318802, 88.3256683, -188.0525665, 188.0480957
19: -64.9723663, 41.1811447, -64.9765778, 41.1886978, -106.1610565, 106.1577148
20: -69.4813843, 53.1743393, -69.4863281, 53.1781311, -122.6595154, 122.6606598
21: -84.8355408, 53.7853394, -84.8407974, 53.8054314, -138.6409607, 138.6261292
22: -88.9120178, 52.7000275, -88.9172287, 52.7086220, -141.6206360, 141.6172485
23: -67.5456543, 57.7058563, -67.5504608, 57.7137070, -125.2593613, 125.2563171
24: -88.2593384, 66.0144196, -88.2648621, 66.0227737, -154.2821045, 154.2792664
25: -77.9002457, 60.1505051, -77.9055023, 60.1576233, -138.0578613, 138.0559998
26: -109.3970261, 90.6831818, -109.4038696, 90.6977692, -200.0947723, 200.0870514
27: -88.0051117, 66.4275208, -88.0110931, 66.4353485, -154.4404602, 154.4386139
28: -69.1465912, 62.4998474, -69.1503906, 62.5054321, -131.6520233, 131.6502380
29: -90.2856140, 47.3662109, -90.2915268, 47.3705902, -137.6561890, 137.6577301
30: -89.2205734, 74.5660553, -89.2276306, 74.5717468, -163.7923279, 163.7936859
31: -87.1298294, 64.0869904, -87.1347809, 64.1023483, -151.2321777, 151.2217712
32: -100.1118546, 64.6576691, -100.1363602, 64.6618347, -164.7736816, 164.7940369
33: -133.0290833, 83.3148499, -133.0374451, 83.3190918, -216.3481750, 216.3522949
34: -117.4673386, 62.7784157, -117.4777527, 62.7821426, -180.2494812, 180.2561646
35: -108.4649048, 71.5093231, -108.4708939, 71.5144653, -179.9793701, 179.9802246
36: -111.1997452, 70.1994781, -111.2063904, 70.2035904, -181.4033203, 181.4058685
37: -152.0449677, 71.0758362, -152.0548401, 71.0831757, -223.1281433, 223.1306610
38: -133.2972717, 84.8709869, -133.3044739, 84.8773193, -218.1745911, 218.1754608
39: -149.1843262, 87.7836761, -149.1922302, 87.7890930, -236.9734192, 236.9759064
40: -115.5243454, 66.5658417, -115.5497513, 66.5696793, -182.0940247, 182.1156006
41: -105.0462570, 75.5291519, -105.0542831, 75.5330887, -180.5793457, 180.5834351
42: -76.4584351, 57.2741852, -76.4830627, 57.2775574, -133.7359924, 133.7572479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=526, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=719, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1623

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8572715, upper bound: 85.9066369
time: 147.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8572715, upper bound: 85.9009616
time: 256.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 407.18 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 407.18
Output dim: 19, lower bound: -85.8572715, upper bound: 85.8866068
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 407.18
Output dim: 19, lower bound: -85.8572715, upper bound: 85.8810289
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 407.18
Output dim: 19, lower bound: -85.8572715, upper bound: 85.9066369
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 407.18
Output dim: 19, lower bound: -85.8572715, upper bound: 85.9009616

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -132.0539551, 77.5912857, -132.1861877, 77.6578369, -209.7117920, 209.7774506
1: -67.3035583, 55.9300041, -67.3470459, 55.9934959, -123.2970505, 123.2770538
2: -56.9794769, 60.2776337, -57.0433540, 60.3075371, -117.2870178, 117.3209839
3: -73.7386169, 69.9691925, -73.8907318, 70.1223755, -143.8609924, 143.8599243
4: -73.5483704, 69.5599747, -73.6667404, 69.6335526, -143.1819153, 143.2267151
5: -69.7127838, 72.0768509, -69.8567505, 72.1883392, -141.9011230, 141.9335938
6: -100.1716843, 73.2526398, -100.1606216, 73.2247314, -173.3964233, 173.4132385
7: -84.3766479, 67.5118408, -84.4577484, 67.5443726, -151.9210205, 151.9695892
8: -101.5568848, 86.9790955, -101.6721191, 87.0470276, -188.6039124, 188.6512146
9: -71.8181305, 72.2530289, -71.8942642, 72.4394226, -144.2575531, 144.1472931
10: -95.8763733, 87.6313477, -96.0451965, 87.8199005, -183.6962738, 183.6765137
11: -90.1565094, 58.6515427, -90.3132324, 58.7002525, -148.8567505, 148.9647675
12: -104.6959457, 89.6006088, -104.6034851, 89.6675720, -194.3634796, 194.2041016
13: -111.9279251, 98.8562698, -111.8999634, 98.9705963, -210.8984985, 210.7562103
14: -160.3039551, 76.4404907, -160.4024963, 76.3793945, -236.6833496, 236.8429871
15: -80.0326996, 66.5060196, -80.1347122, 66.6727066, -146.7053986, 146.6407166
16: -99.3287201, 71.4782410, -99.4119415, 71.5598297, -170.8885498, 170.8901367
17: -157.6551208, 74.3734894, -157.7884369, 74.2935257, -231.9486389, 232.1619110
18: -99.5088272, 88.0630646, -99.6324615, 88.0046463, -187.5134583, 187.6955109
19: -64.7590179, 41.0623741, -64.9198456, 41.0634270, -105.8224487, 105.9822235
20: -69.3130188, 53.0589447, -69.4179535, 53.0731544, -122.3861694, 122.4768982
21: -84.6246262, 53.6633682, -84.7632141, 53.6693840, -138.2940063, 138.4265747
22: -88.7520065, 52.5928459, -88.8272858, 52.6080055, -141.3600159, 141.4201355
23: -67.2809753, 57.4397011, -67.4899597, 57.5347328, -124.8156967, 124.9296417
24: -87.9850616, 65.6945267, -88.1778412, 65.7337646, -153.7188110, 153.8723450
25: -77.6650696, 59.9099541, -77.8403168, 59.9536057, -137.6186676, 137.7502594
26: -109.1976013, 90.5008240, -109.2957916, 90.5446930, -199.7422791, 199.7966156
27: -87.7864075, 66.1195984, -87.9236832, 66.1751480, -153.9615479, 154.0432739
28: -68.9374542, 62.2732773, -69.0952759, 62.3320847, -131.2695312, 131.3685608
29: -90.1086273, 47.2370491, -90.1895905, 47.2809410, -137.3895569, 137.4266357
30: -88.9957581, 74.3077240, -89.1434174, 74.3395386, -163.3352966, 163.4511414
31: -86.8541260, 63.9517670, -87.0545349, 63.9259949, -150.7801208, 151.0062866
32: -99.9322662, 64.6143951, -99.8251495, 64.5949478, -164.5272217, 164.4395447
33: -132.8066864, 83.2435760, -132.8643494, 83.2649689, -216.0716400, 216.1079254
34: -117.3423843, 62.7047653, -117.3754349, 62.7338600, -180.0762482, 180.0802002
35: -108.3020325, 71.3742752, -108.3558655, 71.4521408, -179.7541504, 179.7301331
36: -111.0179214, 70.0391083, -111.0409012, 70.1365356, -181.1544495, 181.0800171
37: -151.7269440, 70.8255539, -151.8693390, 70.9740448, -222.7009888, 222.6948853
38: -133.0711670, 84.6696320, -133.1510925, 84.7767639, -217.8479004, 217.8207245
39: -148.9200287, 87.6711273, -148.9626160, 87.7316971, -236.6517181, 236.6337433
40: -115.3099060, 66.5364990, -115.3473816, 66.5201416, -181.8300323, 181.8838654
41: -104.8646164, 75.4207001, -104.8338089, 75.4700165, -180.3346252, 180.2545166
42: -76.2999649, 57.2355995, -76.2412872, 57.2213936, -133.5213470, 133.4768677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=716, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8135417, upper bound: 85.8037588
time: 146.49 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8116951
time: 169.15 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -132.0628357, 77.5981140, -132.3183746, 77.7578506, -209.8206787, 209.9164886
1: -67.3073959, 55.9365616, -67.4135895, 56.0643539, -123.3717499, 123.3501511
2: -56.9831963, 60.2898979, -57.1271744, 60.4002380, -117.3834229, 117.4170685
3: -73.7421112, 69.9736176, -73.9739227, 70.2012482, -143.9433441, 143.9475403
4: -73.5520477, 69.5734711, -73.7890625, 69.7591095, -143.3111420, 143.3625336
5: -69.7164612, 72.0893402, -69.9638062, 72.3047028, -142.0211487, 142.0531464
6: -100.2062912, 73.2583160, -100.3497849, 73.3887939, -173.5950775, 173.6080933
7: -84.3824997, 67.5286102, -84.5821381, 67.6445465, -152.0270386, 152.1107483
8: -101.5609512, 86.9995575, -101.7821045, 87.1772079, -188.7381592, 188.7816620
9: -71.8430176, 72.2582703, -72.0466156, 72.6226196, -144.4656372, 144.3048859
10: -95.8809128, 87.6425018, -96.1467819, 87.9417496, -183.8226318, 183.7892456
11: -90.1700211, 58.6645546, -90.5435104, 58.8053513, -148.9753723, 149.2080688
12: -104.7514343, 89.6105881, -104.8694687, 89.9313660, -194.6827850, 194.4800568
13: -111.9859085, 98.8634491, -112.1817398, 99.3202972, -211.3062134, 211.0451813
14: -160.3170776, 76.4913635, -160.6572113, 76.5889969, -236.9060669, 237.1485596
15: -80.0379181, 66.5154343, -80.2227936, 66.7736969, -146.8116150, 146.7382202
16: -99.3408661, 71.4900208, -99.6401978, 71.6612015, -171.0020447, 171.1301880
17: -157.6660156, 74.4301453, -158.1403503, 74.5557556, -232.2217712, 232.5704651
18: -99.5162506, 88.1179428, -100.0015869, 88.2733459, -187.7895660, 188.1195221
19: -64.7633286, 41.0843964, -65.0863495, 41.1725922, -105.9359207, 106.1707306
20: -69.3215561, 53.0763130, -69.5362549, 53.1619606, -122.4834976, 122.6125641
21: -84.6328049, 53.6865540, -84.9443130, 53.7845001, -138.4172974, 138.6308594
22: -88.7637558, 52.6049347, -89.0168381, 52.6926422, -141.4563904, 141.6217651
23: -67.2872849, 57.4599113, -67.6441650, 57.6522331, -124.9395142, 125.1040726
24: -87.9966125, 65.7425461, -88.4622574, 65.9569397, -153.9535522, 154.2048035
25: -77.6721725, 59.9413223, -77.9756927, 60.1050682, -137.7772217, 137.9170074
26: -109.2115250, 90.5128174, -109.5673676, 90.6513214, -199.8628387, 200.0801697
27: -87.7984619, 66.1611633, -88.2039108, 66.3671570, -154.1656189, 154.3650818
28: -68.9441223, 62.2981644, -69.2572708, 62.4627075, -131.4068298, 131.5554352
29: -90.1224747, 47.2463684, -90.3589096, 47.3506546, -137.4731140, 137.6052856
30: -89.0082169, 74.3459320, -89.3997955, 74.5327301, -163.5409393, 163.7457275
31: -86.8606491, 63.9879265, -87.2628632, 64.0943909, -150.9550476, 151.2507935
32: -100.0003815, 64.6225281, -100.1341324, 64.8597717, -164.8601379, 164.7566528
33: -132.8359680, 83.2492523, -133.0432892, 83.4584503, -216.2944183, 216.2925415
34: -117.3584900, 62.7063637, -117.5001144, 62.7902603, -180.1487427, 180.2064819
35: -108.3201828, 71.3770752, -108.4811172, 71.5279694, -179.8481445, 179.8581848
36: -111.0446243, 70.0418091, -111.2013855, 70.2749786, -181.3196106, 181.2431946
37: -151.7517395, 70.8309402, -152.0545654, 71.0828857, -222.8346100, 222.8854828
38: -133.0911255, 84.6771393, -133.3038635, 84.9345627, -218.0256805, 217.9810028
39: -148.9531250, 87.6755600, -149.1640167, 87.9502563, -236.9033813, 236.8395691
40: -115.3353882, 66.5405884, -115.5285492, 66.6677475, -182.0031128, 182.0691376
41: -104.9113159, 75.4274292, -105.0558624, 75.6384277, -180.5497437, 180.4832916
42: -76.3513565, 57.2414742, -76.4764557, 57.4993095, -133.8506622, 133.7179260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8135417, upper bound: 85.7975482
time: 173.38 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8052164
time: 213.63 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -132.2913055, 77.7433548, -132.2557983, 77.6793213, -209.9706116, 209.9991455
1: -67.3877716, 56.0491295, -67.3739624, 56.0122604, -123.4000320, 123.4230804
2: -57.0919571, 60.3600807, -57.0798416, 60.3201141, -117.4120712, 117.4399185
3: -73.9963455, 70.1614532, -73.9796371, 70.1389160, -144.1352539, 144.1410828
4: -73.7495499, 69.7300262, -73.7339783, 69.6521759, -143.4017334, 143.4639893
5: -69.9437027, 72.2539520, -69.9338684, 72.2046204, -142.1483154, 142.1878204
6: -100.2747345, 73.2695465, -100.1842117, 73.2508087, -173.5255432, 173.4537506
7: -84.5245819, 67.6145935, -84.5081863, 67.5596313, -152.0841980, 152.1227722
8: -101.7468185, 87.1404953, -101.7365646, 87.0697250, -188.8165283, 188.8770599
9: -72.0474701, 72.4768982, -71.9703751, 72.4581604, -144.5056152, 144.4472656
10: -96.1822128, 87.8782501, -96.1421204, 87.8430634, -184.0252686, 184.0203552
11: -90.3913422, 58.7976418, -90.3528519, 58.7482643, -149.1395874, 149.1504974
12: -104.7839813, 89.7386627, -104.6249542, 89.7079926, -194.4919434, 194.3636017
13: -112.1228409, 99.0319443, -111.9639435, 99.0041885, -211.1270294, 210.9958801
14: -160.5040741, 76.5472641, -160.4589539, 76.4102783, -236.9143219, 237.0062256
15: -80.2222900, 66.7205353, -80.1942902, 66.6934967, -146.9157715, 146.9148102
16: -99.4938812, 71.6260910, -99.4567032, 71.5811234, -171.0749969, 171.0827942
17: -157.8995361, 74.5039825, -157.8624268, 74.3267670, -232.2263031, 232.3664093
18: -99.7098236, 88.2401810, -99.6726990, 88.0659943, -187.7758179, 187.9128723
19: -64.9637604, 41.1552353, -64.9472427, 41.0966644, -106.0604248, 106.1024704
20: -69.4689789, 53.1522064, -69.4427795, 53.1009369, -122.5699158, 122.5949860
21: -84.8227844, 53.7546463, -84.7970657, 53.6999855, -138.5227661, 138.5517120
22: -88.8940811, 52.6787033, -88.8553543, 52.6353912, -141.5294647, 141.5340576
23: -67.5360336, 57.6784859, -67.5171890, 57.6189728, -125.1550064, 125.1956787
24: -88.2410583, 65.9582825, -88.2011719, 65.8276978, -154.0687561, 154.1594238
25: -77.8882904, 60.1115303, -77.8649750, 60.0220032, -137.9102783, 137.9765015
26: -109.3759384, 90.6532669, -109.3308563, 90.5955048, -199.9714203, 199.9841003
27: -87.9863205, 66.3783112, -87.9451065, 66.2653809, -154.2517090, 154.3234253
28: -69.1365509, 62.4690247, -69.1156921, 62.3961334, -131.5326691, 131.5847168
29: -90.2647552, 47.3501511, -90.2197647, 47.3182869, -137.5830383, 137.5699158
30: -89.2034302, 74.5162125, -89.1681442, 74.4046783, -163.6081085, 163.6843567
31: -87.1168365, 64.0446167, -87.0899277, 63.9586792, -151.0755005, 151.1345367
32: -100.0298386, 64.6447144, -99.8490982, 64.6176682, -164.6474915, 164.4938049
33: -132.9876709, 83.3058929, -132.8970642, 83.2885590, -216.2762299, 216.2029419
34: -117.4447174, 62.7718811, -117.3992233, 62.7602234, -180.2049408, 180.1711121
35: -108.4375458, 71.5044632, -108.3784180, 71.4976273, -179.9351807, 179.8828735
36: -111.1583023, 70.1952515, -111.0612717, 70.1892166, -181.3475189, 181.2565002
37: -152.0032501, 71.0679626, -151.9094086, 71.0558014, -223.0590515, 222.9773712
38: -133.2621613, 84.8604736, -133.1826324, 84.8408890, -218.1030579, 218.0430908
39: -149.1286621, 87.7774963, -149.0026093, 87.7680969, -236.8967285, 236.7801056
40: -115.4802704, 66.5594177, -115.3934937, 66.5473938, -182.0276642, 181.9529114
41: -104.9901276, 75.5196762, -104.8576355, 75.5002823, -180.4904175, 180.3773041
42: -76.3963623, 57.2632866, -76.2635651, 57.2399139, -133.6362610, 133.5268555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=718, inp2_unstable=718, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8135417, upper bound: 85.8237173
time: 172.20 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8316159
time: 140.92 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -132.3001709, 77.7501984, -132.3880005, 77.7794037, -210.0795746, 210.1381989
1: -67.3916321, 56.0556755, -67.4405212, 56.0831528, -123.4747849, 123.4962006
2: -57.0956764, 60.3723373, -57.1636391, 60.4128799, -117.5085602, 117.5359650
3: -73.9998322, 70.1658936, -74.0627975, 70.2178040, -144.2176361, 144.2286987
4: -73.7532196, 69.7434921, -73.8562927, 69.7777863, -143.5310059, 143.5997925
5: -69.9473190, 72.2664185, -70.0409088, 72.3209915, -142.2683105, 142.3073273
6: -100.3093109, 73.2752380, -100.3734283, 73.4148407, -173.7241516, 173.6486664
7: -84.5304413, 67.6314011, -84.6325836, 67.6598206, -152.1902466, 152.2639771
8: -101.7508850, 87.1608963, -101.8465958, 87.1998978, -188.9507751, 189.0074921
9: -72.0723572, 72.4821091, -72.1227417, 72.6413422, -144.7136841, 144.6048584
10: -96.1867599, 87.8893738, -96.2436905, 87.9648972, -184.1516571, 184.1330566
11: -90.4048767, 58.8106270, -90.5831604, 58.8533707, -149.2582397, 149.3937836
12: -104.8394623, 89.7486267, -104.8909760, 89.9716949, -194.8111572, 194.6395874
13: -112.1807938, 99.0391617, -112.2457123, 99.3538818, -211.5346527, 211.2848816
14: -160.5171967, 76.5981750, -160.7136536, 76.6198730, -237.1370697, 237.3117981
15: -80.2275085, 66.7299271, -80.2823639, 66.7944641, -147.0219727, 147.0122986
16: -99.5060272, 71.6379089, -99.6849518, 71.6824722, -171.1885071, 171.3228455
17: -157.9104614, 74.5605698, -158.2143250, 74.5889816, -232.4994507, 232.7748718
18: -99.7171631, 88.2950439, -100.0418091, 88.3345718, -188.0517273, 188.3368530
19: -64.9680634, 41.1772614, -65.1137390, 41.2058372, -106.1739044, 106.2909851
20: -69.4775162, 53.1695862, -69.5610352, 53.1897278, -122.6672440, 122.7306213
21: -84.8309784, 53.7778511, -84.9781113, 53.8150940, -138.6460571, 138.7559662
22: -88.9058533, 52.6907959, -89.0449295, 52.7200279, -141.6258698, 141.7357178
23: -67.5423203, 57.6987152, -67.6714172, 57.7364693, -125.2787933, 125.3701248
24: -88.2526169, 66.0063095, -88.4855957, 66.0508957, -154.3035126, 154.4918823
25: -77.8953934, 60.1429062, -78.0003510, 60.1734619, -138.0688477, 138.1432495
26: -109.3898392, 90.6652679, -109.6024551, 90.7021179, -200.0919495, 200.2677307
27: -87.9983521, 66.4198456, -88.2253265, 66.4573822, -154.4557190, 154.6451721
28: -69.1432190, 62.4939003, -69.2776718, 62.5267181, -131.6699219, 131.7715607
29: -90.2786407, 47.3594589, -90.3890762, 47.3879929, -137.6666260, 137.7485352
30: -89.2158661, 74.5544205, -89.4245224, 74.5978165, -163.8136749, 163.9789276
31: -87.1233749, 64.0807648, -87.2982635, 64.1270905, -151.2504578, 151.3790283
32: -100.0979385, 64.6528778, -100.1580887, 64.8824768, -164.9804077, 164.8109589
33: -133.0169373, 83.3116074, -133.0760498, 83.4820480, -216.4989929, 216.3876648
34: -117.4608002, 62.7734642, -117.5239334, 62.8166122, -180.2774048, 180.2973938
35: -108.4556580, 71.5072098, -108.5037384, 71.5734634, -180.0291138, 180.0109253
36: -111.1849976, 70.1979370, -111.2218170, 70.3276672, -181.5126648, 181.4197540
37: -152.0280151, 71.0733871, -152.0946350, 71.1646576, -223.1926727, 223.1680298
38: -133.2821350, 84.8680420, -133.3354492, 84.9986954, -218.2808228, 218.2034607
39: -149.1617584, 87.7820282, -149.2040100, 87.9866486, -237.1483917, 236.9860382
40: -115.5057220, 66.5634918, -115.5745926, 66.6949921, -182.2007141, 182.1380615
41: -105.0368118, 75.5263748, -105.0797272, 75.6686783, -180.7054749, 180.6060944
42: -76.4477692, 57.2691994, -76.4987640, 57.5178070, -133.9655762, 133.7679596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=525, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=719, inp2_unstable=718, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1123
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1217
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1201
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1122
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1137
type: A, layer: 1, pos: 1200
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1184
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1153
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1202
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1168
type: A, layer: 1, pos: 1107
type: A, layer: 1, pos: 1185
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1139
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1274
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1106
type: A, layer: 1, pos: 1216
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1124
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1120
type: A, layer: 1, pos: 1121
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1136
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1169
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1279
type: A, layer: 1, pos: 1186
type: A, layer: 1, pos: 1090
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1026
type: A, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.8135417, upper bound: 85.8174372
time: 163.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8251224
time: 157.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 323.05 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 323.05
Output dim: 19, lower bound: -85.8135417, upper bound: 85.8037588
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 323.05
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8116951
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 323.05
Output dim: 19, lower bound: -85.8135417, upper bound: 85.7975482
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 323.05
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8052164
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 323.05
Output dim: 19, lower bound: -85.8135417, upper bound: 85.8237173
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 323.05
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8316159
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 323.05
Output dim: 19, lower bound: -85.8135417, upper bound: 85.8174372
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 323.05
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8251224

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -131.9320831, 77.5561523, -132.1776733, 77.6553650, -209.5874481, 209.7338257
1: -67.2647552, 55.9079628, -67.3443069, 55.9919395, -123.2566910, 123.2522736
2: -56.9057465, 60.2531891, -57.0382042, 60.3058357, -117.2115555, 117.2913895
3: -73.5864639, 69.9303741, -73.8802261, 70.1196518, -143.7061157, 143.8105927
4: -73.4746857, 69.5246658, -73.6615143, 69.6310883, -143.1057739, 143.1861877
5: -69.5944824, 72.0389099, -69.8485565, 72.1856842, -141.7801666, 141.8874664
6: -100.1206055, 73.2156982, -100.1570740, 73.2221527, -173.3427429, 173.3727722
7: -84.2851257, 67.4766846, -84.4514008, 67.5419312, -151.8270264, 151.9280853
8: -101.4720306, 86.9485474, -101.6660309, 87.0448914, -188.5169067, 188.6145630
9: -71.7133102, 72.2085800, -71.8869781, 72.4363327, -144.1496429, 144.0955505
10: -95.8194427, 87.5652313, -96.0411835, 87.8152924, -183.6347351, 183.6063995
11: -90.0970154, 58.5534744, -90.3090668, 58.6934471, -148.7904663, 148.8625488
12: -104.6384354, 89.3541565, -104.5994873, 89.6505127, -194.2889404, 193.9536438
13: -111.8681870, 98.7827759, -111.8955612, 98.9654922, -210.8336487, 210.6783142
14: -160.2060852, 76.2388611, -160.3956909, 76.3654175, -236.5715027, 236.6345215
15: -79.9429169, 66.4594193, -80.1282806, 66.6694336, -146.6123505, 146.5876923
16: -99.2618866, 71.4446945, -99.4072342, 71.5574799, -170.8193665, 170.8519287
17: -157.5687561, 74.1188049, -157.7824402, 74.2759323, -231.8446808, 231.9012451
18: -99.4460678, 87.8961487, -99.6281128, 87.9928284, -187.4388733, 187.5242615
19: -64.7125397, 41.0449715, -64.9165802, 41.0622253, -105.7747650, 105.9615479
20: -69.2622833, 53.0090408, -69.4143982, 53.0696869, -122.3319473, 122.4234314
21: -84.5685654, 53.6287384, -84.7592926, 53.6669502, -138.2355194, 138.3880310
22: -88.6986923, 52.5511703, -88.8235550, 52.6050720, -141.3037720, 141.3747253
23: -67.2363281, 57.4229622, -67.4868164, 57.5335693, -124.7698975, 124.9097748
24: -87.9338226, 65.6632767, -88.1742554, 65.7315826, -153.6654053, 153.8375244
25: -77.6201553, 59.8779411, -77.8371735, 59.9513741, -137.5715332, 137.7151184
26: -109.1321259, 90.2949219, -109.2911911, 90.5304718, -199.6625977, 199.5860748
27: -87.7165451, 66.0598450, -87.9188232, 66.1709976, -153.8875427, 153.9786682
28: -68.8842468, 62.2479782, -69.0915680, 62.3303452, -131.2145844, 131.3395386
29: -90.0551834, 47.1841125, -90.1859207, 47.2771835, -137.3323669, 137.3700256
30: -88.9432297, 74.2178421, -89.1397476, 74.3330383, -163.2762756, 163.3575897
31: -86.7902145, 63.9260139, -87.0500336, 63.9241829, -150.7143860, 150.9760437
32: -99.8817749, 64.5702667, -99.8216248, 64.5918732, -164.4736328, 164.3918762
33: -132.6721191, 83.1901016, -132.8550110, 83.2612305, -215.9333191, 216.0450897
34: -117.2570648, 62.6714287, -117.3695221, 62.7315140, -179.9885559, 180.0409546
35: -108.2192383, 71.3485413, -108.3500137, 71.4502869, -179.6695251, 179.6985474
36: -110.9743652, 70.0111237, -111.0378571, 70.1345901, -181.1089478, 181.0489807
37: -151.6508179, 70.7748566, -151.8640137, 70.9704895, -222.6213074, 222.6388550
38: -133.0049744, 84.6136093, -133.1464844, 84.7727890, -217.7777710, 217.7600861
39: -148.8517151, 87.6352844, -148.9578094, 87.7292099, -236.5809326, 236.5930939
40: -115.2380524, 66.5173645, -115.3423920, 66.5187836, -181.7568359, 181.8597565
41: -104.8029404, 75.3905182, -104.8295288, 75.4678955, -180.2708435, 180.2200470
42: -76.2404099, 57.1994934, -76.2371597, 57.2188721, -133.4592896, 133.4366455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=716, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 19, lower bound: -85.7616769, upper bound: 85.7956394
time: 173.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 19, lower bound: -85.7616769, upper bound: 85.7956394
time: 148.36 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -132.1378174, 77.9160767, -132.1629181, 77.6518936, -209.7896729, 210.0789948
1: -67.3506012, 56.0772247, -67.3385468, 55.9883614, -123.3389587, 123.4157639
2: -57.0059509, 60.5630875, -57.0322762, 60.3018112, -117.3077621, 117.5953674
3: -73.7774811, 70.3521118, -73.8763885, 70.1128998, -143.8903809, 144.2285004
4: -73.5772095, 69.8392181, -73.6539841, 69.6283340, -143.2055359, 143.4931946
5: -69.7396088, 72.4376831, -69.8404541, 72.1803436, -141.9199524, 142.2781372
6: -100.4218369, 73.3278503, -100.1527710, 73.2067261, -173.6285553, 173.4806213
7: -84.4472656, 67.6917572, -84.4447479, 67.5355148, -151.9827728, 152.1364899
8: -101.6025314, 87.1663284, -101.6592178, 87.0390930, -188.6416321, 188.8255463
9: -71.8845520, 72.5927963, -71.8804169, 72.4317169, -144.3162689, 144.4732056
10: -96.0369110, 87.9329376, -96.0347519, 87.8092804, -183.8461914, 183.9676666
11: -90.7025528, 58.6399193, -90.3040543, 58.6702728, -149.3728027, 148.9439697
12: -105.2413940, 89.6403809, -104.5928345, 89.6428833, -194.8842468, 194.2332153
13: -111.9360809, 99.1027527, -111.8706589, 98.9589310, -210.8950043, 210.9734192
14: -160.8825836, 76.4688568, -160.3864136, 76.3596497, -237.2422333, 236.8552551
15: -80.0649643, 66.7508545, -80.1061707, 66.6656036, -146.7305603, 146.8570251
16: -99.5215225, 71.6776428, -99.4004517, 71.5503845, -171.0718994, 171.0780945
17: -158.4418030, 74.4189529, -157.7748413, 74.2694092, -232.7112122, 232.1937866
18: -99.9787979, 88.0718231, -99.6181183, 87.9678650, -187.9466553, 187.6899414
19: -65.0570221, 41.1528244, -64.9107361, 41.0605583, -106.1175766, 106.0635452
20: -69.5911560, 53.0844307, -69.4099655, 53.0590057, -122.6501617, 122.4943848
21: -85.0552597, 53.6904716, -84.7534332, 53.6550179, -138.7102814, 138.4439087
22: -89.0677948, 52.6647644, -88.8146362, 52.6017265, -141.6695251, 141.4794006
23: -67.5291061, 57.5283585, -67.4821472, 57.5294571, -125.0585403, 125.0104904
24: -88.2508316, 65.7334442, -88.1644440, 65.7228012, -153.9736328, 153.8978882
25: -77.8093262, 59.9801826, -77.8280411, 59.9484177, -137.7577515, 137.8082275
26: -109.8314209, 90.5249481, -109.2807312, 90.5239410, -200.3553619, 199.8056641
27: -88.2009811, 66.0988846, -87.9113235, 66.1452789, -154.3462524, 154.0101929
28: -69.2128143, 62.3238220, -69.0875397, 62.3264656, -131.5392761, 131.4113464
29: -90.5692291, 47.2746887, -90.1782684, 47.2729568, -137.8421936, 137.4529572
30: -89.3322372, 74.3255997, -89.1338806, 74.3124542, -163.6446838, 163.4594727
31: -87.0920563, 64.0424042, -87.0425873, 63.9222260, -151.0142822, 151.0849915
32: -100.0889130, 64.6924133, -99.8165054, 64.5857544, -164.6746521, 164.5089111
33: -132.8741455, 83.7516785, -132.8449097, 83.2577209, -216.1318665, 216.5965576
34: -117.4234924, 63.0559235, -117.3621140, 62.7275696, -180.1510468, 180.4180298
35: -108.3477859, 71.6274414, -108.3332214, 71.4467621, -179.7945404, 179.9606628
36: -111.2699738, 70.1212845, -111.0323334, 70.1324081, -181.4023743, 181.1536255
37: -152.0218506, 70.8691406, -151.8530884, 70.9583511, -222.9801941, 222.7222290
38: -133.3007355, 84.7821808, -133.1368103, 84.7620850, -218.0628204, 217.9189758
39: -149.0043030, 88.0792542, -148.9430695, 87.7268677, -236.7311707, 237.0223083
40: -115.4664078, 66.6423950, -115.3355331, 66.5149384, -181.9812927, 181.9779358
41: -105.0064468, 75.5456009, -104.8251648, 75.4631348, -180.4695740, 180.3707581
42: -76.4175415, 57.3925705, -76.2340393, 57.2129326, -133.6304626, 133.6266174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.7304839, upper bound: 85.8033391
time: 181.24 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 19, lower bound: -85.7304839, upper bound: 85.8033391
time: 191.73 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -131.9406128, 77.5627975, -132.3098602, 77.7553787, -209.6959839, 209.8726501
1: -67.2684326, 55.9144592, -67.4108582, 56.0627975, -123.3312302, 123.3253174
2: -56.9093208, 60.2652664, -57.1220551, 60.3985100, -117.3078156, 117.3873215
3: -73.5896301, 69.9346848, -73.9634018, 70.1985474, -143.7881775, 143.8980865
4: -73.4779205, 69.5379944, -73.7838745, 69.7566528, -143.2345734, 143.3218384
5: -69.5980453, 72.0511398, -69.9556122, 72.3020401, -141.9000854, 142.0067444
6: -100.1550217, 73.2212372, -100.3462219, 73.3862152, -173.5412140, 173.5674591
7: -84.2908630, 67.4933472, -84.5757904, 67.6421051, -151.9329681, 152.0691376
8: -101.4759903, 86.9687347, -101.7760315, 87.1750412, -188.6510315, 188.7447510
9: -71.7372894, 72.2136459, -72.0392990, 72.6195221, -144.3567963, 144.2529449
10: -95.8238220, 87.5761108, -96.1427460, 87.9371719, -183.7609863, 183.7188568
11: -90.1102905, 58.5658607, -90.5393677, 58.7985458, -148.9088440, 149.1052246
12: -104.6936035, 89.3640213, -104.8654633, 89.9142151, -194.6078033, 194.2294922
13: -111.9246674, 98.7898178, -112.1773605, 99.3152008, -211.2398682, 210.9671631
14: -160.2187195, 76.2883606, -160.6504211, 76.5750198, -236.7937317, 236.9387817
15: -79.9480591, 66.4687500, -80.2163467, 66.7704468, -146.7185059, 146.6850891
16: -99.2736969, 71.4563065, -99.6355515, 71.6588287, -170.9325256, 171.0918579
17: -157.5791016, 74.1741867, -158.1343689, 74.5381393, -232.1172028, 232.3085632
18: -99.4532394, 87.9498901, -99.9972534, 88.2614822, -187.7146759, 187.9471436
19: -64.7167053, 41.0668793, -65.0830917, 41.1713943, -105.8880997, 106.1499634
20: -69.2706757, 53.0262070, -69.5326843, 53.1584969, -122.4291534, 122.5588913
21: -84.5764618, 53.6514206, -84.9403915, 53.7820473, -138.3585052, 138.5917969
22: -88.7101974, 52.5631027, -89.0131226, 52.6897202, -141.3999176, 141.5762329
23: -67.2425003, 57.4426613, -67.6410217, 57.6510620, -124.8935623, 125.0836792
24: -87.9452286, 65.7105179, -88.4586868, 65.9547653, -153.8999939, 154.1692047
25: -77.6270599, 59.9087219, -77.9725647, 60.1028252, -137.7298889, 137.8812866
26: -109.1457748, 90.3062286, -109.5628357, 90.6370850, -199.7828674, 199.8690643
27: -87.7283554, 66.1003876, -88.1990738, 66.3629913, -154.0913391, 154.2994385
28: -68.8907623, 62.2725372, -69.2535553, 62.4609604, -131.3517151, 131.5260925
29: -90.0687408, 47.1926422, -90.3552170, 47.3469200, -137.4156647, 137.5478516
30: -88.9554520, 74.2549591, -89.3961639, 74.5262146, -163.4816589, 163.6511078
31: -86.7965317, 63.9618492, -87.2583771, 64.0925903, -150.8891144, 151.2202301
32: -99.9489746, 64.5782471, -100.1306305, 64.8567200, -164.8056946, 164.7088776
33: -132.7003174, 83.1955872, -133.0339813, 83.4547272, -216.1550140, 216.2295532
34: -117.2727814, 62.6728249, -117.4941864, 62.7879105, -180.0606842, 180.1670074
35: -108.2367554, 71.3512421, -108.4752960, 71.5261459, -179.7628784, 179.8265381
36: -111.0006027, 70.0137558, -111.1983490, 70.2730408, -181.2736511, 181.2120972
37: -151.6752319, 70.7801208, -152.0492249, 71.0793457, -222.7545471, 222.8293457
38: -133.0245361, 84.6208572, -133.2992401, 84.9306183, -217.9551544, 217.9201050
39: -148.8833313, 87.6396179, -149.1592407, 87.9477921, -236.8311157, 236.7988586
40: -115.2631989, 66.5213928, -115.5235138, 66.6664124, -181.9296112, 182.0449066
41: -104.8493271, 75.3970642, -105.0515442, 75.6362991, -180.4856262, 180.4486084
42: -76.2916718, 57.2052803, -76.4723358, 57.4967957, -133.7884674, 133.6776123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=716, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 19, lower bound: -85.7616769, upper bound: 85.7890839
time: 176.89 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 19, lower bound: -85.7616769, upper bound: 85.7890839
time: 267.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -132.1463318, 77.9225845, -132.2951202, 77.7518692, -209.8981781, 210.2177124
1: -67.3542786, 56.0836792, -67.4050903, 56.0591927, -123.4134674, 123.4887695
2: -57.0095520, 60.5750694, -57.1160965, 60.3944817, -117.4040375, 117.6911621
3: -73.7806396, 70.3563766, -73.9595795, 70.1917953, -143.9724274, 144.3159485
4: -73.5804291, 69.8524323, -73.7763062, 69.7539139, -143.3343506, 143.6287384
5: -69.7431793, 72.4497833, -69.9475250, 72.2967529, -142.0399170, 142.3973083
6: -100.4561768, 73.3333893, -100.3419571, 73.3708801, -173.8270569, 173.6753540
7: -84.4530106, 67.7084808, -84.5691376, 67.6358261, -152.0888367, 152.2776184
8: -101.6064758, 87.1864243, -101.7691956, 87.1692581, -188.7757111, 188.9555969
9: -71.9084778, 72.5978622, -72.0327148, 72.6148682, -144.5233459, 144.6305847
10: -96.0411530, 87.9438248, -96.1362839, 87.9311676, -183.9723053, 184.0801086
11: -90.7157364, 58.6522865, -90.5343857, 58.7754326, -149.4911652, 149.1866760
12: -105.2965851, 89.6503830, -104.8589783, 89.9066849, -195.2032776, 194.5093384
13: -111.9926529, 99.1097565, -112.1525116, 99.3085556, -211.3011627, 211.2622528
14: -160.8951263, 76.5182571, -160.6411438, 76.5692444, -237.4643707, 237.1593933
15: -80.0701447, 66.7600708, -80.1942368, 66.7666168, -146.8367615, 146.9543152
16: -99.5333862, 71.6892548, -99.6287842, 71.6518631, -171.1852417, 171.3180237
17: -158.4521179, 74.4743042, -158.1267548, 74.5316162, -232.9837341, 232.6010590
18: -99.9859314, 88.1254730, -99.9871445, 88.2365341, -188.2224731, 188.1126099
19: -65.0611877, 41.1747284, -65.0772552, 41.1697388, -106.2309265, 106.2519760
20: -69.5995331, 53.1015854, -69.5282440, 53.1478004, -122.7473297, 122.6298294
21: -85.0631256, 53.7131462, -84.9345322, 53.7701225, -138.8332520, 138.6476746
22: -89.0793152, 52.6766167, -89.0041809, 52.6863708, -141.7656860, 141.6807861
23: -67.5352783, 57.5480843, -67.6363678, 57.6469612, -125.1822357, 125.1844482
24: -88.2621841, 65.7806778, -88.4488449, 65.9460220, -154.2081909, 154.2295227
25: -77.8162308, 60.0109367, -77.9634399, 60.0998764, -137.9161072, 137.9743805
26: -109.8450546, 90.5362015, -109.5523453, 90.6305466, -200.4756012, 200.0885468
27: -88.2127380, 66.1393585, -88.1914062, 66.3372650, -154.5499878, 154.3307648
28: -69.2193298, 62.3484077, -69.2495575, 62.4571495, -131.6764832, 131.5979614
29: -90.5828323, 47.2831345, -90.3476105, 47.3426704, -137.9254913, 137.6307373
30: -89.3442993, 74.3630676, -89.3903122, 74.5060120, -163.8503113, 163.7533875
31: -87.0983734, 64.0782242, -87.2508850, 64.0905762, -151.1889343, 151.3291016
32: -100.1560898, 64.7003784, -100.1255569, 64.8506546, -165.0067444, 164.8259125
33: -132.9024963, 83.7571564, -133.0238495, 83.4512405, -216.3537292, 216.7809753
34: -117.4393311, 63.0572510, -117.4867554, 62.7839317, -180.2232666, 180.5440063
35: -108.3653183, 71.6300659, -108.4584656, 71.5226135, -179.8879089, 180.0885315
36: -111.2962570, 70.1240005, -111.1929321, 70.2708740, -181.5671387, 181.3169250
37: -152.0462494, 70.8744202, -152.0382996, 71.0672531, -223.1134644, 222.9127045
38: -133.3202820, 84.7896042, -133.2898102, 84.9199219, -218.2402039, 218.0793915
39: -149.0359802, 88.0835724, -149.1445312, 87.9454651, -236.9814148, 237.2280884
40: -115.4915390, 66.6464462, -115.5166473, 66.6625366, -182.1540833, 182.1630859
41: -105.0527573, 75.5521240, -105.0472183, 75.6315613, -180.6843262, 180.5993195
42: -76.4687653, 57.3983269, -76.4692459, 57.4908409, -133.9596100, 133.8675690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=525, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=717, inp2_unstable=716, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1123
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1217
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1201
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1122
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1137
type: B, layer: 1, pos: 1200
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1184
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1153
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1202
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1168
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1107
type: B, layer: 1, pos: 1185
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1139
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1274
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1106
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1216
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1124
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1120
type: B, layer: 1, pos: 1121
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1136
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1169
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 1279
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1186
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1090
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1026
type: B, layer: 1, pos: 836

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 637

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 19, lower bound: -85.7304839, upper bound: 85.7966168
time: 147.75 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 19, lower bound: -85.7304839, upper bound: 85.7966168
time: 154.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 304.66 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 304.66
Output dim: 19, lower bound: -85.7616769, upper bound: 85.7956394
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 304.66
Output dim: 19, lower bound: -85.7616769, upper bound: 85.7956394
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 304.66
Output dim: 19, lower bound: -85.7304839, upper bound: 85.8033391
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 304.66
Output dim: 19, lower bound: -85.7304839, upper bound: 85.8033391
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 304.66
Output dim: 19, lower bound: -85.7616769, upper bound: 85.7890839
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 304.66
Output dim: 19, lower bound: -85.7616769, upper bound: 85.7890839
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 304.66
Output dim: 19, lower bound: -85.7304839, upper bound: 85.7966168
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 304.66
Output dim: 19, lower bound: -85.7304839, upper bound: 85.7966168
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 304.66
Output dim: 19, lower bound: -85.8135417, upper bound: 85.8237173
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 304.66
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8316159
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 304.66
Output dim: 19, lower bound: -85.8135417, upper bound: 85.8174372
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 304.66
Output dim: 19, lower bound: -85.7819112, upper bound: 85.8251224
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=106.16907501220703
rel_dist={19: [-85.9497628281792, 85.94976282669441]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 12496.52 seconds
