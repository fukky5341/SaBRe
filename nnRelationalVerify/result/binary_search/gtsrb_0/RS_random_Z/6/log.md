## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 120.087095659
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062)
1: (-66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460)
2: (-58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640)
3: (-67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885)
4: (-75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559)
5: (-65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220)
6: (-109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789)
7: (-78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697)
8: (-87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605)
9: (-75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662)
10: (-107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393)
11: (-103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957)
12: (-102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781)
13: (-110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346)
14: (-160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135)
15: (-88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279)
16: (-109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046)
17: (-156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399)
18: (-105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549)
19: (-78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546)
20: (-75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198)
21: (-98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444)
22: (-100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295)
23: (-78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809)
24: (-96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657)
25: (-83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241)
26: (-114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608)
27: (-98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791)
28: (-77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825)
29: (-104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177)
30: (-98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945)
31: (-103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775)
32: (-108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908)
33: (-135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370)
34: (-111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219)
35: (-111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279)
36: (-114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459)
37: (-159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663)
38: (-135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160)
39: (-151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490)
40: (-124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654)
41: (-108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199)
42: (-79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359)

## BASE Result
execution time: IAR + LP analysis = 2.91 + 179.84 = 182.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -127.4375409, upper bound: 127.4375409


# Binary Search by BASE starts (time budget: 17817.25 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=187.93016052246094
rel_dist={8: [-120.19734924920294, 120.19734925085426]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=187.93016052246094
rel_dist={8: [-114.51163586720978, 114.51163585533283]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=187.93016052246094
rel_dist={8: [-116.6081349540645, 116.60813496216502]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=187.93016052246094
rel_dist={8: [-118.48656646956891, 118.48656646223236]}

## Binary Search Result
Binary search time: 1837.84 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 15979.41 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1618

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4439213, upper bound: 124.4482411
time: 220.05 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4482411, upper bound: 124.4439213
time: 139.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 359.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 359.82
Output dim: 8, lower bound: -124.4439213, upper bound: 124.4482411
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 359.82
Output dim: 8, lower bound: -124.4482411, upper bound: 124.4439213

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 606

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 523

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4432711, upper bound: 124.4433680
time: 117.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4390442, upper bound: 124.4475986
time: 136.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1696

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1567

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4402591, upper bound: 124.4435635
time: 148.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4478811, upper bound: 124.4359297
time: 175.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 325.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 325.54
Output dim: 8, lower bound: -124.4432711, upper bound: 124.4433680
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 325.54
Output dim: 8, lower bound: -124.4390442, upper bound: 124.4475986
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 325.54
Output dim: 8, lower bound: -124.4402591, upper bound: 124.4435635
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 325.54
Output dim: 8, lower bound: -124.4478811, upper bound: 124.4359297

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1646

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1720

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2925184, upper bound: 124.2928424
time: 865.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2925184, upper bound: 124.2928424
time: 862.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 737

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3745241, upper bound: 124.4451390
time: 129.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4365925, upper bound: 124.3830318
time: 155.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 771

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 658

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4251397, upper bound: 124.4376325
time: 130.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4344167, upper bound: 124.4285848
time: 110.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 781

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1580

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4471437, upper bound: 124.4320959
time: 117.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4441614, upper bound: 124.4352033
time: 1326.17 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 1446.19 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1446.19
Output dim: 8, lower bound: -124.2925184, upper bound: 124.2928424
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1446.19
Output dim: 8, lower bound: -124.2925184, upper bound: 124.2928424
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1446.19
Output dim: 8, lower bound: -124.3745241, upper bound: 124.4451390
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1446.19
Output dim: 8, lower bound: -124.4365925, upper bound: 124.3830318
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1446.19
Output dim: 8, lower bound: -124.4251397, upper bound: 124.4376325
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1446.19
Output dim: 8, lower bound: -124.4344167, upper bound: 124.4285848
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1446.19
Output dim: 8, lower bound: -124.4471437, upper bound: 124.4320959
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1446.19
Output dim: 8, lower bound: -124.4441614, upper bound: 124.4352033
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=187.93016052246094
rel_dist={8: [-124.44835295992456, 124.44835294926614]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1712

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 771

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7714000, upper bound: 121.7734898
time: 134.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7734898, upper bound: 121.7713999
time: 117.45 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 251.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 251.83
Output dim: 8, lower bound: -121.7714000, upper bound: 121.7734898
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 251.83
Output dim: 8, lower bound: -121.7734898, upper bound: 121.7713999

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1607

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 725

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7527983, upper bound: 121.7548851
time: 133.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7527983, upper bound: 121.7548851
time: 131.20 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 857

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 705

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7720298, upper bound: 121.6787136
time: 125.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6807825, upper bound: 121.7699415
time: 124.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 251.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 251.99
Output dim: 8, lower bound: -121.7527983, upper bound: 121.7548851
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 251.99
Output dim: 8, lower bound: -121.7527983, upper bound: 121.7548851
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 251.99
Output dim: 8, lower bound: -121.7720298, upper bound: 121.6787136
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 251.99
Output dim: 8, lower bound: -121.6807825, upper bound: 121.7699415

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1546

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1569

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7527905, upper bound: 121.7548816
time: 135.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7527976, upper bound: 121.7548746
time: 141.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 877

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7497260, upper bound: 121.7518221
time: 166.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7497260, upper bound: 121.7540389
time: 1316.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1608

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7614274, upper bound: 121.6777470
time: 159.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7710584, upper bound: 121.6682702
time: 114.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 530

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6770988, upper bound: 121.7693611
time: 106.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6802025, upper bound: 121.7661767
time: 141.92 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 250.58 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 250.58
Output dim: 8, lower bound: -121.7527905, upper bound: 121.7548816
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 250.58
Output dim: 8, lower bound: -121.7527976, upper bound: 121.7548746
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 250.58
Output dim: 8, lower bound: -121.7497260, upper bound: 121.7518221
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 250.58
Output dim: 8, lower bound: -121.7497260, upper bound: 121.7540389
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 250.58
Output dim: 8, lower bound: -121.7614274, upper bound: 121.6777470
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 250.58
Output dim: 8, lower bound: -121.7710584, upper bound: 121.6682702
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 250.58
Output dim: 8, lower bound: -121.6770988, upper bound: 121.7693611
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 250.58
Output dim: 8, lower bound: -121.6802025, upper bound: 121.7661767

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1780

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 931

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7511300, upper bound: 121.7142573
time: 185.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7121560, upper bound: 121.7532452
time: 174.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1435

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7488550, upper bound: 121.7499295
time: 133.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7478464, upper bound: 121.7509395
time: 124.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 824

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1402

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7383767, upper bound: 121.7505673
time: 109.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7506988, upper bound: 121.7382621
time: 203.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 315.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 315.92
Output dim: 8, lower bound: -121.7511300, upper bound: 121.7142573
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 315.92
Output dim: 8, lower bound: -121.7121560, upper bound: 121.7532452
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 315.92
Output dim: 8, lower bound: -121.7488550, upper bound: 121.7499295
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 315.92
Output dim: 8, lower bound: -121.7478464, upper bound: 121.7509395
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 315.92
Output dim: 8, lower bound: -121.7383767, upper bound: 121.7505673
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 315.92
Output dim: 8, lower bound: -121.7506988, upper bound: 121.7382621
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 315.92
Output dim: 8, lower bound: -121.7497260, upper bound: 121.7540389
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 315.92
Output dim: 8, lower bound: -121.7614274, upper bound: 121.6777470
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 315.92
Output dim: 8, lower bound: -121.7710584, upper bound: 121.6682702
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 315.92
Output dim: 8, lower bound: -121.6770988, upper bound: 121.7693611
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 315.92
Output dim: 8, lower bound: -121.6802025, upper bound: 121.7661767
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=187.93016052246094
rel_dist={8: [-121.77351111190006, 121.77351109658267]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1556

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 608

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1968421, upper bound: 120.1875152
time: 110.52 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1875152, upper bound: 120.1968421
time: 118.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 228.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 228.85
Output dim: 8, lower bound: -120.1968421, upper bound: 120.1875152
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 228.85
Output dim: 8, lower bound: -120.1875152, upper bound: 120.1968421

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 610

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 748

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1652664, upper bound: 120.1559330
time: 153.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1652664, upper bound: 120.1559330
time: 2821.11 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1638

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 940

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1850187, upper bound: 120.1965014
time: 113.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1871700, upper bound: 120.1944206
time: 154.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 270.79 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 270.79
Output dim: 8, lower bound: -120.1652664, upper bound: 120.1559330
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 270.79
Output dim: 8, lower bound: -120.1652664, upper bound: 120.1559330
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 270.79
Output dim: 8, lower bound: -120.1850187, upper bound: 120.1965014
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 270.79
Output dim: 8, lower bound: -120.1871700, upper bound: 120.1944206

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1449

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 947

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1638403, upper bound: 120.0909748
time: 175.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1002693, upper bound: 120.1545154
time: 123.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -127.2149048, 89.1640396, -127.2149048, 89.1640396, -216.3789368, 216.3789062
1: -66.5396576, 68.0918961, -66.5396576, 68.0918961, -134.6315613, 134.6315460
2: -58.5631943, 69.2564697, -58.5631943, 69.2564697, -127.8196640, 127.8196640
3: -67.7209244, 78.7972870, -67.7209244, 78.7972870, -146.5181885, 146.5181885
4: -75.2664948, 81.6539688, -75.2664948, 81.6539688, -156.9204712, 156.9204559
5: -65.5062637, 76.1869659, -65.5062637, 76.1869659, -141.6932068, 141.6932220
6: -109.8069153, 78.5995789, -109.8069153, 78.5995789, -188.4064941, 188.4064789
7: -78.8454895, 72.8652802, -78.8454895, 72.8652802, -151.7107697, 151.7107697
8: -87.1600952, 100.7700653, -87.1600952, 100.7700653, -187.9301605, 187.9301605
9: -75.7390900, 75.6362762, -75.7390900, 75.6362762, -151.3753662, 151.3753662
10: -107.4068985, 101.2758331, -107.4068985, 101.2758331, -208.6827393, 208.6827393
11: -103.4638443, 64.2717590, -103.4638443, 64.2717590, -167.7355957, 167.7355957
12: -102.0476837, 80.4811172, -102.0476837, 80.4811172, -182.5287781, 182.5287781
13: -110.8701401, 108.0016251, -110.8701401, 108.0016251, -218.8717346, 218.8717346
14: -160.6092529, 91.1176605, -160.6092529, 91.1176605, -251.7268982, 251.7269135
15: -88.5221710, 76.5145569, -88.5221710, 76.5145569, -165.0367126, 165.0367279
16: -109.9994507, 78.5158691, -109.9994507, 78.5158691, -188.5153198, 188.5153046
17: -156.5166779, 84.0092926, -156.5166779, 84.0092926, -240.5259247, 240.5259399
18: -105.0823746, 77.3176956, -105.0823746, 77.3176956, -182.4000549, 182.4000549
19: -78.0874557, 51.5048065, -78.0874557, 51.5048065, -129.5922546, 129.5922546
20: -75.5377579, 56.6650543, -75.5377579, 56.6650543, -132.2028198, 132.2028198
21: -98.2792664, 59.5399704, -98.2792664, 59.5399704, -157.8192291, 157.8192444
22: -100.2264175, 61.8101311, -100.2264175, 61.8101311, -162.0365295, 162.0365295
23: -78.4364395, 66.0052490, -78.4364395, 66.0052490, -144.4416809, 144.4416809
24: -96.8337479, 67.0911331, -96.8337479, 67.0911331, -163.9248810, 163.9248657
25: -83.8780975, 68.2253265, -83.8780975, 68.2253265, -152.1034241, 152.1034241
26: -114.8598328, 88.0768280, -114.8598328, 88.0768280, -202.9366608, 202.9366608
27: -98.5206451, 68.5798340, -98.5206451, 68.5798340, -167.1004791, 167.1004791
28: -77.7635345, 67.3654633, -77.7635345, 67.3654633, -145.1289978, 145.1289825
29: -104.0128326, 58.4327927, -104.0128326, 58.4327927, -162.4456177, 162.4456177
30: -98.2490463, 75.2553406, -98.2490463, 75.2553406, -173.5043945, 173.5043945
31: -103.1321411, 70.8435364, -103.1321411, 70.8435364, -173.9756775, 173.9756775
32: -108.8962402, 69.1468506, -108.8962402, 69.1468506, -178.0430908, 178.0430908
33: -135.0666199, 93.2563095, -135.0666199, 93.2563095, -228.3229370, 228.3229370
34: -111.5981445, 68.8105927, -111.5981445, 68.8105927, -180.4087219, 180.4087219
35: -111.4385529, 74.4787827, -111.4385529, 74.4787827, -185.9173279, 185.9173279
36: -114.8498077, 73.8041382, -114.8498077, 73.8041382, -188.6539459, 188.6539459
37: -159.7998047, 73.2518692, -159.7998047, 73.2518692, -233.0516663, 233.0516663
38: -135.1685486, 86.5441208, -135.1685486, 86.5441208, -221.7126160, 221.7126160
39: -151.2247162, 90.3598328, -151.2247162, 90.3598328, -241.5845490, 241.5845490
40: -124.3997498, 69.2173080, -124.3997498, 69.2173080, -193.6170349, 193.6170654
41: -108.5153732, 79.3525543, -108.5153732, 79.3525543, -187.8679199, 187.8679199
42: -79.3159485, 69.0089874, -79.3159485, 69.0089874, -148.3249359, 148.3249359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=524, inp2_unstable=524, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 593

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1645343, upper bound: 120.1300222
time: 106.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1300222, upper bound: 120.1551988
time: 1510.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 1619.24 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1619.24
Output dim: 8, lower bound: -120.1638403, upper bound: 120.0909748
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1619.24
Output dim: 8, lower bound: -120.1002693, upper bound: 120.1545154
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1619.24
Output dim: 8, lower bound: -120.1645343, upper bound: 120.1300222
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1619.24
Output dim: 8, lower bound: -120.1300222, upper bound: 120.1551988
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1619.24
Output dim: 8, lower bound: -120.1850187, upper bound: 120.1965014
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1619.24
Output dim: 8, lower bound: -120.1871700, upper bound: 120.1944206
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=187.93016052246094
rel_dist={8: [-120.19734924920294, 120.19734925085426]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 14527.28 seconds
