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
execution time: IAR + LP analysis = 2.91 + 181.15 = 184.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -127.4375409, upper bound: 127.4375409


# Binary Search by BASE starts (time budget: 17815.94 seconds, max iter: 100)

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
Binary search time: 1837.16 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 15978.78 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1785

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4200285, upper bound: 124.3116487
time: 460.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4198911, upper bound: 124.4198909
time: 156.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 617.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 617.33
Output dim: 8, lower bound: -124.4200285, upper bound: 124.3116487
IS_A2, status: Status.UNKNOWN, split count: 1, time: 617.33
Output dim: 8, lower bound: -124.4198911, upper bound: 124.4198909

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -127.1311264, 89.1436310, -127.2077103, 89.1623077, -216.2934265, 216.3513489
1: -66.5138092, 68.0384521, -66.5374680, 68.0873260, -134.6011353, 134.5758972
2: -58.5468636, 69.1890488, -58.5617828, 69.2506943, -127.7975388, 127.7508316
3: -67.7071228, 78.7096252, -67.7197266, 78.7896729, -146.4967957, 146.4293518
4: -75.2487946, 81.5308990, -75.2649918, 81.6434326, -156.8922272, 156.7958984
5: -65.4886627, 76.1151276, -65.5047607, 76.1807709, -141.6694336, 141.6198730
6: -109.7483292, 78.5794678, -109.8019028, 78.5978928, -188.3462067, 188.3813782
7: -78.8203201, 72.7568512, -78.8433685, 72.8560867, -151.6764069, 151.6002197
8: -87.1449585, 100.5661545, -87.1588135, 100.7528915, -187.8978577, 187.7249603
9: -75.7243347, 75.5817566, -75.7378082, 75.6316986, -151.3560181, 151.3195648
10: -107.3807678, 101.2442780, -107.4047012, 101.2731628, -208.6539307, 208.6489716
11: -103.3352585, 64.2569122, -103.4529266, 64.2705231, -167.6057739, 167.7098236
12: -101.9506149, 80.4471130, -102.0394135, 80.4782104, -182.4288330, 182.4865265
13: -110.8471680, 107.9141769, -110.8681717, 107.9941177, -218.8412781, 218.7823486
14: -160.5510559, 91.0945892, -160.6043091, 91.1157227, -251.6667786, 251.6988983
15: -88.4957047, 76.4759598, -88.5199585, 76.5112762, -165.0069733, 164.9959106
16: -109.9002151, 78.5042496, -109.9909058, 78.5148926, -188.4151001, 188.4951477
17: -156.3343353, 83.9851913, -156.5014343, 84.0072479, -240.3415833, 240.4865875
18: -104.9205704, 77.3037720, -105.0687485, 77.3165207, -182.2370911, 182.3725281
19: -78.0098114, 51.4939690, -78.0807190, 51.5038757, -129.5136871, 129.5746918
20: -75.5152817, 56.6305389, -75.5358582, 56.6621361, -132.1773987, 132.1663971
21: -98.2035141, 59.5246239, -98.2728424, 59.5386887, -157.7422028, 157.7974701
22: -100.1405792, 61.8021851, -100.2191010, 61.8094864, -161.9500732, 162.0212708
23: -78.3402786, 65.9895554, -78.4281769, 66.0038986, -144.3441772, 144.4177246
24: -96.7298279, 67.0822754, -96.8249283, 67.0903625, -163.8201904, 163.9071960
25: -83.8112106, 68.2155304, -83.8724136, 68.2245026, -152.0357056, 152.0879517
26: -114.8033524, 88.0512085, -114.8550644, 88.0746231, -202.8779755, 202.9062653
27: -98.4599075, 68.5677719, -98.5154343, 68.5787964, -167.0386963, 167.0831909
28: -77.7048492, 67.3536377, -77.7585602, 67.3644485, -145.0692902, 145.1121979
29: -103.8900681, 58.4268112, -104.0022202, 58.4322815, -162.3223267, 162.4290314
30: -98.1680603, 75.2357712, -98.2421417, 75.2536926, -173.4217529, 173.4779053
31: -103.0106888, 70.8320160, -103.1216736, 70.8425598, -173.8532410, 173.9536896
32: -108.8633575, 69.1159210, -108.8934021, 69.1442642, -178.0075989, 178.0093231
33: -135.0333557, 93.2313156, -135.0637817, 93.2541962, -228.2875519, 228.2951050
34: -111.4851456, 68.7869720, -111.5886383, 68.8086090, -180.2937622, 180.3755951
35: -111.3274078, 74.4565659, -111.4292297, 74.4769135, -185.8043213, 185.8858032
36: -114.7816772, 73.7921448, -114.8439941, 73.8031616, -188.5848236, 188.6361389
37: -159.6750336, 73.2374573, -159.7891541, 73.2506332, -232.9256592, 233.0266113
38: -135.1294861, 86.5081635, -135.1652832, 86.5410614, -221.6705322, 221.6734314
39: -151.1862946, 90.2730103, -151.2214661, 90.3524933, -241.5387878, 241.4944458
40: -124.3573074, 69.1886826, -124.3961334, 69.2148590, -193.5721436, 193.5847931
41: -108.4479294, 79.3339767, -108.5096436, 79.3509521, -187.7988892, 187.8436279
42: -79.2953491, 68.9645996, -79.3141937, 69.0052490, -148.3005829, 148.2787933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=524, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3116487, upper bound: 124.3116487
time: 104.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3116487, upper bound: 124.3116487
time: 131.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -127.3538971, 89.2551880, -127.2087097, 89.1632462, -216.5171356, 216.4638977
1: -66.6497192, 68.1133728, -66.5387650, 68.0901642, -134.7398834, 134.6521301
2: -58.6951294, 69.2781219, -58.5626221, 69.2540894, -127.9492188, 127.8407440
3: -67.8862610, 78.8332672, -67.7203217, 78.7950287, -146.6812744, 146.5535889
4: -75.4861374, 81.6803513, -75.2658844, 81.6505966, -157.1367188, 156.9462280
5: -65.5944061, 76.2170715, -65.5055695, 76.1849823, -141.7793884, 141.7226410
6: -109.8595734, 78.7055054, -109.8050232, 78.5990524, -188.4586182, 188.5105133
7: -78.9908905, 72.8946762, -78.8442917, 72.8631744, -151.8540497, 151.7389679
8: -87.5101776, 100.8090134, -87.1594772, 100.7660599, -188.2762451, 187.9684906
9: -75.8492584, 75.6747284, -75.7385483, 75.6349640, -151.4842224, 151.4132690
10: -107.4923935, 101.3588181, -107.4059677, 101.2750168, -208.7673950, 208.7647705
11: -103.5003204, 64.4044418, -103.4586716, 64.2712021, -167.7715149, 167.8631134
12: -102.0819778, 80.6744537, -102.0453873, 80.4801483, -182.5621033, 182.7198181
13: -111.0131760, 108.0716629, -110.8694229, 107.9986725, -219.0118256, 218.9410858
14: -160.7323914, 91.2356033, -160.6043854, 91.1171036, -251.8494873, 251.8399811
15: -88.6518555, 76.5433502, -88.5213852, 76.5129929, -165.1648560, 165.0647278
16: -110.0696335, 78.6890869, -109.9966354, 78.5155563, -188.5851898, 188.6857300
17: -156.6102295, 84.2303314, -156.5123138, 84.0086060, -240.6188354, 240.7426147
18: -105.1457748, 77.6100159, -105.0790787, 77.3172684, -182.4630280, 182.6890869
19: -78.1299133, 51.5710678, -78.0849915, 51.5045204, -129.6344147, 129.6560516
20: -75.6346283, 56.7000542, -75.5369873, 56.6603203, -132.2949524, 132.2370453
21: -98.3277283, 59.6157417, -98.2767258, 59.5395889, -157.8673096, 157.8924713
22: -100.3006821, 61.9020538, -100.2244415, 61.8096504, -162.1103363, 162.1264954
23: -78.4733353, 66.1574326, -78.4337921, 66.0047531, -144.4780884, 144.5912170
24: -96.9062805, 67.2642975, -96.8316727, 67.0907898, -163.9970703, 164.0959473
25: -83.9311523, 68.3194427, -83.8765106, 68.2247467, -152.1558838, 152.1959381
26: -114.9255447, 88.2026978, -114.8581543, 88.0761108, -203.0016479, 203.0608368
27: -98.6040497, 68.6654434, -98.5188065, 68.5795135, -167.1835632, 167.1842499
28: -77.8043823, 67.4609833, -77.7615356, 67.3650284, -145.1694031, 145.2225037
29: -104.1020508, 58.5877991, -104.0102310, 58.4324074, -162.5344543, 162.5980225
30: -98.3139038, 75.4237823, -98.2464676, 75.2546616, -173.5685730, 173.6702423
31: -103.1951599, 71.0233078, -103.1288910, 70.8431473, -174.0382996, 174.1521912
32: -108.9527740, 69.2304688, -108.8952255, 69.1461945, -178.0989685, 178.1256866
33: -135.1309052, 93.3098450, -135.0656433, 93.2549210, -228.3858337, 228.3754883
34: -111.6538925, 68.9791489, -111.5958710, 68.8097076, -180.4635925, 180.5750122
35: -111.4924774, 74.6328888, -111.4363480, 74.4779053, -185.9703827, 186.0692291
36: -114.8892288, 73.8942490, -114.8482513, 73.8036499, -188.6928711, 188.7424927
37: -159.8751068, 73.4185944, -159.7963867, 73.2511444, -233.1262360, 233.2149811
38: -135.2311096, 86.6103363, -135.1674805, 86.5418930, -221.7730103, 221.7778015
39: -151.3312378, 90.3884811, -151.2235870, 90.3575897, -241.6888123, 241.6120605
40: -124.4881058, 69.2769165, -124.3985367, 69.2154617, -193.7035675, 193.6754456
41: -108.5610733, 79.4581070, -108.5131760, 79.3520432, -187.9131012, 187.9712830
42: -79.3646469, 69.0527725, -79.3151550, 69.0055695, -148.3702087, 148.3679199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=524, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3059522, upper bound: 124.4016393
time: 157.19 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3059522, upper bound: 124.4016393
time: 147.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 306.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 306.62
Output dim: 8, lower bound: -124.3116487, upper bound: 124.3116487
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 306.62
Output dim: 8, lower bound: -124.3116487, upper bound: 124.3116487
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 306.62
Output dim: 8, lower bound: -124.3059522, upper bound: 124.4016393
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 306.62
Output dim: 8, lower bound: -124.3059522, upper bound: 124.4016393

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -127.1311264, 89.1436310, -127.1311264, 89.1436310, -216.2747498, 216.2747498
1: -66.5138092, 68.0384521, -66.5138092, 68.0384521, -134.5522614, 134.5522461
2: -58.5468636, 69.1890488, -58.5468636, 69.1890488, -127.7359161, 127.7359161
3: -67.7071228, 78.7096252, -67.7071228, 78.7096252, -146.4167480, 146.4167480
4: -75.2487946, 81.5308990, -75.2487946, 81.5308990, -156.7796936, 156.7796936
5: -65.4886627, 76.1151276, -65.4886627, 76.1151276, -141.6037903, 141.6037903
6: -109.7483292, 78.5794678, -109.7483292, 78.5794678, -188.3277893, 188.3277893
7: -78.8203201, 72.7568512, -78.8203201, 72.7568512, -151.5771790, 151.5771790
8: -87.1449585, 100.5661545, -87.1449585, 100.5661545, -187.7110901, 187.7111053
9: -75.7243347, 75.5817566, -75.7243347, 75.5817566, -151.3060760, 151.3060913
10: -107.3807678, 101.2442780, -107.3807678, 101.2442780, -208.6250153, 208.6250153
11: -103.3352585, 64.2569122, -103.3352585, 64.2569122, -167.5921631, 167.5921631
12: -101.9506149, 80.4471130, -101.9506149, 80.4471130, -182.3977356, 182.3977051
13: -110.8471680, 107.9141769, -110.8471680, 107.9141769, -218.7613525, 218.7613525
14: -160.5510559, 91.0945892, -160.5510559, 91.0945892, -251.6456299, 251.6456299
15: -88.4957047, 76.4759598, -88.4957047, 76.4759598, -164.9716644, 164.9716644
16: -109.9002151, 78.5042496, -109.9002151, 78.5042496, -188.4044495, 188.4044647
17: -156.3343353, 83.9851913, -156.3343353, 83.9851913, -240.3195190, 240.3194885
18: -104.9205704, 77.3037720, -104.9205704, 77.3037720, -182.2243347, 182.2243347
19: -78.0098114, 51.4939690, -78.0098114, 51.4939690, -129.5037842, 129.5037842
20: -75.5152817, 56.6305389, -75.5152817, 56.6305389, -132.1458130, 132.1458130
21: -98.2035141, 59.5246239, -98.2035141, 59.5246239, -157.7281342, 157.7281342
22: -100.1405792, 61.8021851, -100.1405792, 61.8021851, -161.9427643, 161.9427643
23: -78.3402786, 65.9895554, -78.3402786, 65.9895554, -144.3298035, 144.3298035
24: -96.7298279, 67.0822754, -96.7298279, 67.0822754, -163.8121033, 163.8121033
25: -83.8112106, 68.2155304, -83.8112106, 68.2155304, -152.0267334, 152.0267334
26: -114.8033524, 88.0512085, -114.8033524, 88.0512085, -202.8545532, 202.8545532
27: -98.4599075, 68.5677719, -98.4599075, 68.5677719, -167.0276642, 167.0276794
28: -77.7048492, 67.3536377, -77.7048492, 67.3536377, -145.0584717, 145.0584869
29: -103.8900681, 58.4268112, -103.8900681, 58.4268112, -162.3168793, 162.3168640
30: -98.1680603, 75.2357712, -98.1680603, 75.2357712, -173.4038086, 173.4038086
31: -103.0106888, 70.8320160, -103.0106888, 70.8320160, -173.8427124, 173.8427124
32: -108.8633575, 69.1159210, -108.8633575, 69.1159210, -177.9792480, 177.9792633
33: -135.0333557, 93.2313156, -135.0333557, 93.2313156, -228.2646790, 228.2646790
34: -111.4851456, 68.7869720, -111.4851456, 68.7869720, -180.2721252, 180.2721252
35: -111.3274078, 74.4565659, -111.3274078, 74.4565659, -185.7839661, 185.7839661
36: -114.7816772, 73.7921448, -114.7816772, 73.7921448, -188.5737915, 188.5738220
37: -159.6750336, 73.2374573, -159.6750336, 73.2374573, -232.9124603, 232.9124756
38: -135.1294861, 86.5081635, -135.1294861, 86.5081635, -221.6376343, 221.6376495
39: -151.1862946, 90.2730103, -151.1862946, 90.2730103, -241.4593048, 241.4592896
40: -124.3573074, 69.1886826, -124.3573074, 69.1886826, -193.5459900, 193.5459595
41: -108.4479294, 79.3339767, -108.4479294, 79.3339767, -187.7819061, 187.7819061
42: -79.2953491, 68.9645996, -79.2953491, 68.9645996, -148.2599487, 148.2599487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1784

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3653658, upper bound: 124.1972172
time: 119.54 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3653658, upper bound: 124.2917852
time: 117.40 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -127.1311264, 89.1436310, -127.3538971, 89.2551880, -216.3863068, 216.4975281
1: -66.5138092, 68.0384521, -66.6497192, 68.1133728, -134.6271667, 134.6881714
2: -58.5468636, 69.1890488, -58.6951294, 69.2781219, -127.8249741, 127.8841782
3: -67.7071228, 78.7096252, -67.8862610, 78.8332672, -146.5403900, 146.5958862
4: -75.2487946, 81.5308990, -75.4861374, 81.6803513, -156.9291382, 157.0170288
5: -65.4886627, 76.1151276, -65.5944061, 76.2170715, -141.7057343, 141.7095337
6: -109.7483292, 78.5794678, -109.8595734, 78.7055054, -188.4538269, 188.4390411
7: -78.8203201, 72.7568512, -78.9908905, 72.8946762, -151.7149963, 151.7477417
8: -87.1449585, 100.5661545, -87.5101776, 100.8090134, -187.9539490, 188.0763245
9: -75.7243347, 75.5817566, -75.8492584, 75.6747284, -151.3990326, 151.4310150
10: -107.3807678, 101.2442780, -107.4923935, 101.3588181, -208.7395630, 208.7366638
11: -103.3352585, 64.2569122, -103.5003204, 64.4044418, -167.7397003, 167.7572327
12: -101.9506149, 80.4471130, -102.0819778, 80.6744537, -182.6250610, 182.5290833
13: -110.8471680, 107.9141769, -111.0131760, 108.0716629, -218.9188232, 218.9273529
14: -160.5510559, 91.0945892, -160.7323914, 91.2356033, -251.7866516, 251.8269806
15: -88.4957047, 76.4759598, -88.6518555, 76.5433502, -165.0390320, 165.1278076
16: -109.9002151, 78.5042496, -110.0696335, 78.6890869, -188.5892944, 188.5738831
17: -156.3343353, 83.9851913, -156.6102295, 84.2303314, -240.5646667, 240.5953827
18: -104.9205704, 77.3037720, -105.1457748, 77.6100159, -182.5305786, 182.4495544
19: -78.0098114, 51.4939690, -78.1299133, 51.5710678, -129.5808716, 129.6238861
20: -75.5152817, 56.6305389, -75.6346283, 56.7000542, -132.2153320, 132.2651672
21: -98.2035141, 59.5246239, -98.3277283, 59.6157417, -157.8192596, 157.8523560
22: -100.1405792, 61.8021851, -100.3006821, 61.9020538, -162.0426331, 162.1028748
23: -78.3402786, 65.9895554, -78.4733353, 66.1574326, -144.4976959, 144.4628754
24: -96.7298279, 67.0822754, -96.9062805, 67.2642975, -163.9941101, 163.9885406
25: -83.8112106, 68.2155304, -83.9311523, 68.3194427, -152.1306458, 152.1466827
26: -114.8033524, 88.0512085, -114.9255447, 88.2026978, -203.0060425, 202.9767456
27: -98.4599075, 68.5677719, -98.6040497, 68.6654434, -167.1253510, 167.1718140
28: -77.7048492, 67.3536377, -77.8043823, 67.4609833, -145.1658325, 145.1580200
29: -103.8900681, 58.4268112, -104.1020508, 58.5877991, -162.4778595, 162.5288696
30: -98.1680603, 75.2357712, -98.3139038, 75.4237823, -173.5918274, 173.5496674
31: -103.0106888, 70.8320160, -103.1951599, 71.0233078, -174.0339813, 174.0271759
32: -108.8633575, 69.1159210, -108.9527740, 69.2304688, -178.0937958, 178.0686951
33: -135.0333557, 93.2313156, -135.1309052, 93.3098450, -228.3432007, 228.3622131
34: -111.4851456, 68.7869720, -111.6538925, 68.9791489, -180.4642944, 180.4408569
35: -111.3274078, 74.4565659, -111.4924774, 74.6328888, -185.9602966, 185.9490356
36: -114.7816772, 73.7921448, -114.8892288, 73.8942490, -188.6759033, 188.6813660
37: -159.6750336, 73.2374573, -159.8751068, 73.4185944, -233.0936127, 233.1125183
38: -135.1294861, 86.5081635, -135.2311096, 86.6103363, -221.7398224, 221.7392731
39: -151.1862946, 90.2730103, -151.3312378, 90.3884811, -241.5747681, 241.6042480
40: -124.3573074, 69.1886826, -124.4881058, 69.2769165, -193.6342163, 193.6767578
41: -108.4479294, 79.3339767, -108.5610733, 79.4581070, -187.9060211, 187.8950500
42: -79.2953491, 68.9645996, -79.3646469, 69.0527725, -148.3481140, 148.3292542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1784

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3653658, upper bound: 124.1972172
time: 238.46 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3653658, upper bound: 124.2917852
time: 144.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -127.3460846, 89.2525864, -127.1503677, 89.1432953, -216.4893799, 216.4029388
1: -66.6468582, 68.1047058, -66.5169449, 68.0260620, -134.6729126, 134.6216431
2: -58.6931877, 69.2686310, -58.5483398, 69.1807251, -127.8739166, 127.8169708
3: -67.8846512, 78.8214874, -67.7081299, 78.7053909, -146.5900421, 146.5296021
4: -75.4840851, 81.6639938, -75.2504120, 81.5253754, -157.0094604, 156.9143982
5: -65.5923309, 76.2062073, -65.4897766, 76.1037369, -141.6960754, 141.6959839
6: -109.8507843, 78.7030334, -109.7388000, 78.5800171, -188.4308014, 188.4418182
7: -78.9882660, 72.8818054, -78.8239365, 72.7643661, -151.7526245, 151.7057495
8: -87.5084839, 100.7829742, -87.1466370, 100.5688553, -188.0773315, 187.9296112
9: -75.8472366, 75.6656799, -75.7232819, 75.5652924, -151.4125061, 151.3889618
10: -107.4892502, 101.3548355, -107.3820190, 101.2449265, -208.7341766, 208.7368469
11: -103.4828415, 64.4025421, -103.3220215, 64.2569351, -167.7397766, 167.7245636
12: -102.0716553, 80.6703568, -101.9676437, 80.4493942, -182.5210571, 182.6380005
13: -111.0104294, 108.0555954, -110.8485565, 107.8782043, -218.8885956, 218.9041443
14: -160.7269287, 91.2327576, -160.5631866, 91.0951157, -251.8220367, 251.7959442
15: -88.6489105, 76.5355377, -88.4990005, 76.4532013, -165.1021118, 165.0345154
16: -110.0595322, 78.6875229, -109.9205246, 78.5034637, -188.5629730, 188.6080475
17: -156.5947266, 84.2277145, -156.3928528, 83.9893036, -240.5839996, 240.6205750
18: -105.1236801, 77.6083374, -104.9083099, 77.3046036, -182.4282532, 182.5166473
19: -78.1185684, 51.5698776, -77.9978790, 51.4957199, -129.6142731, 129.5677490
20: -75.6313858, 56.6976662, -75.5121002, 56.6424065, -132.2737885, 132.2097626
21: -98.3153992, 59.6139908, -98.1829453, 59.5260963, -157.8414917, 157.7969360
22: -100.2909851, 61.9011612, -100.1509094, 61.8030701, -162.0940552, 162.0520630
23: -78.4590378, 66.1554108, -78.3230972, 65.9893799, -144.4484253, 144.4785004
24: -96.8907471, 67.2632446, -96.7130661, 67.0828476, -163.9735870, 163.9763031
25: -83.9212189, 68.3182297, -83.8003540, 68.2156830, -152.1369019, 152.1185760
26: -114.9157410, 88.1997833, -114.7836151, 88.0543823, -202.9701080, 202.9833679
27: -98.5946121, 68.6638260, -98.4465485, 68.5672607, -167.1618652, 167.1103821
28: -77.7946472, 67.4594498, -77.6869354, 67.3533859, -145.1480408, 145.1463928
29: -104.0892410, 58.5869522, -103.9122696, 58.4259186, -162.5151672, 162.4992218
30: -98.3010864, 75.4212952, -98.1485901, 75.2355652, -173.5366364, 173.5698853
31: -103.1763458, 71.0220184, -102.9863129, 70.8333893, -174.0097046, 174.0083313
32: -108.9480591, 69.2265472, -108.8590851, 69.1160889, -178.0641479, 178.0856171
33: -135.1248169, 93.3072662, -135.0193024, 93.2353592, -228.3601685, 228.3265686
34: -111.6396942, 68.9766464, -111.4863281, 68.7902451, -180.4299316, 180.4629517
35: -111.4795914, 74.6304550, -111.3370590, 74.4592896, -185.9388733, 185.9674988
36: -114.8818665, 73.8928757, -114.7931824, 73.7932816, -188.6751404, 188.6860657
37: -159.8599091, 73.4169464, -159.6789551, 73.2386093, -233.0985107, 233.0958862
38: -135.2252808, 86.6066437, -135.1232300, 86.5137177, -221.7389984, 221.7298584
39: -151.3263855, 90.3800888, -151.1865234, 90.2944336, -241.6208038, 241.5665894
40: -124.4824219, 69.2741165, -124.3549805, 69.1942444, -193.6766663, 193.6290894
41: -108.5523911, 79.4556122, -108.4482346, 79.3328247, -187.8852234, 187.9038391
42: -79.3619232, 69.0493622, -79.2943115, 68.9796066, -148.3415222, 148.3436737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2973633, upper bound: 124.2604710
time: 123.39 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2973633, upper bound: 124.3929445
time: 131.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -127.3522339, 89.2547760, -127.3187561, 89.2044449, -216.5566711, 216.5735321
1: -66.6492004, 68.1120911, -66.6390839, 68.1030884, -134.7522888, 134.7511597
2: -58.6947899, 69.2756729, -58.6775780, 69.2641296, -127.9589233, 127.9532471
3: -67.8859177, 78.8316956, -67.8494415, 78.8140717, -146.6999817, 146.6811218
4: -75.4858627, 81.6782455, -75.4450684, 81.6622467, -157.1481018, 157.1233063
5: -65.5940552, 76.2159271, -65.5928650, 76.2024078, -141.7964478, 141.8087921
6: -109.8577194, 78.7051773, -109.8311462, 78.6975555, -188.5552673, 188.5363159
7: -78.9903030, 72.8933105, -78.9556046, 72.8782959, -151.8685913, 151.8489075
8: -87.5099487, 100.8062897, -87.4157791, 100.7849121, -188.2948608, 188.2220459
9: -75.8489685, 75.6737366, -75.8426361, 75.6542969, -151.5032654, 151.5163727
10: -107.4918518, 101.3581467, -107.4837494, 101.3247299, -208.8165741, 208.8418884
11: -103.4974899, 64.4041748, -103.4785690, 64.4157486, -167.9132385, 167.8827362
12: -102.0807343, 80.6738968, -102.0660324, 80.6105499, -182.6912842, 182.7399292
13: -111.0127563, 108.0699692, -111.0248718, 108.0414352, -219.0541687, 219.0948486
14: -160.7316895, 91.2351837, -160.6998138, 91.1804123, -251.9121094, 251.9349976
15: -88.6514587, 76.5417023, -88.6444550, 76.5257187, -165.1771698, 165.1861420
16: -110.0678711, 78.6888580, -110.0376663, 78.6246414, -188.6925049, 188.7265320
17: -156.6074829, 84.2299042, -156.5731201, 84.1289749, -240.7364502, 240.8030090
18: -105.1434708, 77.6097565, -105.1185303, 77.5593033, -182.7027740, 182.7282867
19: -78.1274338, 51.5708389, -78.1034775, 51.5761108, -129.7035522, 129.6743164
20: -75.6341934, 56.6997833, -75.5784302, 56.7048454, -132.3390198, 132.2782135
21: -98.3249969, 59.6155434, -98.3007507, 59.6285744, -157.9535675, 157.9162903
22: -100.2991486, 61.9019165, -100.2686768, 61.8863945, -162.1855469, 162.1705933
23: -78.4715805, 66.1571808, -78.4519501, 66.1477127, -144.6192932, 144.6091309
24: -96.9046478, 67.2640839, -96.8730011, 67.2422333, -164.1468658, 164.1370697
25: -83.9300537, 68.3192444, -83.9091034, 68.3147736, -152.2448120, 152.2283478
26: -114.9238586, 88.2023010, -114.8942108, 88.1992493, -203.1231079, 203.0965118
27: -98.6023560, 68.6652679, -98.5630188, 68.6719971, -167.2743225, 167.2282867
28: -77.8028717, 67.4607697, -77.7801514, 67.4661331, -145.2689972, 145.2409058
29: -104.0992126, 58.5876999, -104.0590668, 58.5499001, -162.6491089, 162.6467590
30: -98.3119278, 75.4234619, -98.2890244, 75.4077301, -173.7196655, 173.7124634
31: -103.1927032, 71.0231094, -103.1624908, 71.0080109, -174.2007141, 174.1855927
32: -108.9518356, 69.2300415, -108.9303284, 69.2171097, -178.1689301, 178.1603699
33: -135.1299133, 93.3093185, -135.1074829, 93.3061523, -228.4360657, 228.4168091
34: -111.6523743, 68.9785767, -111.6311417, 68.9526367, -180.6050110, 180.6097107
35: -111.4911118, 74.6324081, -111.4685669, 74.5962067, -186.0873108, 186.1009827
36: -114.8882599, 73.8939056, -114.8705521, 73.8760681, -188.7643280, 188.7644653
37: -159.8722382, 73.4182358, -159.8410339, 73.3845367, -233.2567596, 233.2592773
38: -135.2301636, 86.6096191, -135.2025452, 86.5988922, -221.8290405, 221.8121643
39: -151.3305969, 90.3869476, -151.2911682, 90.3706207, -241.7012024, 241.6781158
40: -124.4870224, 69.2763290, -124.4498367, 69.2737885, -193.7608032, 193.7261658
41: -108.5595169, 79.4577713, -108.5354996, 79.4454193, -188.0049438, 187.9932556
42: -79.3641891, 69.0519104, -79.3434906, 69.0451584, -148.4093475, 148.3954010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2973633, upper bound: 124.2604688
time: 179.21 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2973633, upper bound: 124.3929445
time: 129.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 311.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 311.72
Output dim: 8, lower bound: -124.3653658, upper bound: 124.1972172
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 311.72
Output dim: 8, lower bound: -124.3653658, upper bound: 124.2917852
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 311.72
Output dim: 8, lower bound: -124.3653658, upper bound: 124.1972172
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 311.72
Output dim: 8, lower bound: -124.3653658, upper bound: 124.2917852
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 311.72
Output dim: 8, lower bound: -124.2973633, upper bound: 124.2604710
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 311.72
Output dim: 8, lower bound: -124.2973633, upper bound: 124.3929445
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 311.72
Output dim: 8, lower bound: -124.2973633, upper bound: 124.2604688
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 311.72
Output dim: 8, lower bound: -124.2973633, upper bound: 124.3929445

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -127.0734787, 89.1233521, -127.1233826, 89.1409760, -216.2144470, 216.2467041
1: -66.4920731, 67.9733582, -66.5109253, 68.0296021, -134.5216675, 134.4842834
2: -58.5326500, 69.1145859, -58.5449486, 69.1793976, -127.7120361, 127.6595154
3: -67.6948395, 78.6202698, -67.7054901, 78.6975250, -146.3923645, 146.3257446
4: -75.2332306, 81.4057465, -75.2467346, 81.5143204, -156.7475586, 156.6524811
5: -65.4728088, 76.0339508, -65.4865723, 76.1041489, -141.5769653, 141.5205078
6: -109.6821823, 78.5604553, -109.7395172, 78.5769501, -188.2591248, 188.2999725
7: -78.7999191, 72.6581268, -78.8176041, 72.7439728, -151.5438843, 151.4757385
8: -87.1319962, 100.3693085, -87.1432343, 100.5401306, -187.6721191, 187.5125427
9: -75.7090302, 75.5121155, -75.7222748, 75.5727081, -151.2817383, 151.2343903
10: -107.3567886, 101.2140045, -107.3775787, 101.2402954, -208.5970764, 208.5915833
11: -103.1987305, 64.2425842, -103.3172379, 64.2549973, -167.4537354, 167.5598145
12: -101.8729324, 80.4162064, -101.9402161, 80.4430237, -182.3159485, 182.3564148
13: -110.8262939, 107.7939148, -110.8443756, 107.8981247, -218.7244263, 218.6382904
14: -160.5100708, 91.0723267, -160.5456238, 91.0917130, -251.6017609, 251.6179504
15: -88.4732742, 76.4162445, -88.4927521, 76.4681396, -164.9414062, 164.9089966
16: -109.8235016, 78.4921341, -109.8899765, 78.5026398, -188.3261414, 188.3821106
17: -156.2149658, 83.9659042, -156.3186340, 83.9826202, -240.1975708, 240.2845459
18: -104.7499847, 77.2910690, -104.8984833, 77.3020782, -182.0520630, 182.1895447
19: -77.9209595, 51.4852257, -77.9981995, 51.4928284, -129.4137726, 129.4834290
20: -75.4902954, 56.6127663, -75.5119934, 56.6281815, -132.1184692, 132.1247559
21: -98.1087189, 59.5111580, -98.1909637, 59.5228462, -157.6315613, 157.7021179
22: -100.0668411, 61.7955933, -100.1307373, 61.8013191, -161.8681641, 161.9263306
23: -78.2297363, 65.9741592, -78.3257828, 65.9875259, -144.2172546, 144.2999268
24: -96.6117249, 67.0742645, -96.7143555, 67.0811920, -163.6929169, 163.7886200
25: -83.7351913, 68.2064438, -83.8012695, 68.2143402, -151.9495239, 152.0077209
26: -114.7287674, 88.0296631, -114.7934875, 88.0483246, -202.7770996, 202.8231506
27: -98.3874283, 68.5555267, -98.4503784, 68.5661163, -166.9535522, 167.0059052
28: -77.6301422, 67.3420258, -77.6950302, 67.3520966, -144.9822388, 145.0370483
29: -103.7922516, 58.4202652, -103.8769913, 58.4259338, -162.2181702, 162.2972565
30: -98.0702820, 75.2165833, -98.1551819, 75.2332306, -173.3035126, 173.3717651
31: -102.8680573, 70.8223114, -102.9916382, 70.8307190, -173.6987762, 173.8139496
32: -108.8271637, 69.0857697, -108.8585968, 69.1120148, -177.9391479, 177.9443665
33: -134.9869385, 93.2117004, -135.0271912, 93.2287445, -228.2156830, 228.2388916
34: -111.3756943, 68.7674942, -111.4709167, 68.7843933, -180.1600800, 180.2384033
35: -111.2282715, 74.4379349, -111.3145065, 74.4541016, -185.6823578, 185.7524414
36: -114.7265778, 73.7817535, -114.7742462, 73.7907944, -188.5173645, 188.5559845
37: -159.5566101, 73.2248459, -159.6595764, 73.2357941, -232.7924042, 232.8844147
38: -135.0851440, 86.4801178, -135.1236420, 86.5044785, -221.5896149, 221.6037445
39: -151.1491089, 90.2098923, -151.1813660, 90.2645340, -241.4136353, 241.3912659
40: -124.3137817, 69.1675568, -124.3515854, 69.1858749, -193.4996643, 193.5191345
41: -108.3826752, 79.3147430, -108.4392319, 79.3314819, -187.7141571, 187.7539673
42: -79.2742310, 68.9386749, -79.2925720, 68.9612122, -148.2354431, 148.2312317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 723

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2240394, upper bound: 124.2541357
time: 133.92 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3571304, upper bound: 124.2541357
time: 123.76 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -127.2407455, 89.1845703, -127.1293793, 89.1432419, -216.3839874, 216.3139496
1: -66.6140671, 68.0520248, -66.5132828, 68.0373535, -134.6514130, 134.5653076
2: -58.6617470, 69.2002640, -58.5465202, 69.1868439, -127.8485870, 127.7467804
3: -67.8363571, 78.7287750, -67.7068481, 78.7083588, -146.5446930, 146.4356232
4: -75.4280472, 81.5427856, -75.2485275, 81.5291061, -156.9571533, 156.7913208
5: -65.5760345, 76.1327362, -65.4882965, 76.1140213, -141.6900330, 141.6210327
6: -109.7745285, 78.6779022, -109.7464752, 78.5791397, -188.3536682, 188.4243774
7: -78.9317245, 72.7721863, -78.8197556, 72.7554932, -151.6872253, 151.5919342
8: -87.4013519, 100.5852280, -87.1447372, 100.5634460, -187.9647980, 187.7299500
9: -75.8284836, 75.6011658, -75.7240601, 75.5808258, -151.4093018, 151.3252106
10: -107.4586868, 101.2938385, -107.3802567, 101.2436600, -208.7023315, 208.6741028
11: -103.3553696, 64.4015198, -103.3329620, 64.2566605, -167.6120300, 167.7344666
12: -101.9713821, 80.5776215, -101.9493942, 80.4465790, -182.4179382, 182.5270081
13: -111.0026932, 107.9570007, -110.8467407, 107.9124985, -218.9151917, 218.8037109
14: -160.6465149, 91.1578064, -160.5502930, 91.0942383, -251.7407074, 251.7080994
15: -88.6188049, 76.4887466, -88.4953461, 76.4743271, -165.0931396, 164.9841003
16: -109.9423828, 78.6133270, -109.8986359, 78.5040131, -188.4463806, 188.5119629
17: -156.3955078, 84.1052170, -156.3318329, 83.9848099, -240.3803101, 240.4370422
18: -104.9601822, 77.5458984, -104.9183044, 77.3035431, -182.2637177, 182.4641876
19: -78.0298309, 51.5655479, -78.0076447, 51.4937553, -129.5235748, 129.5731964
20: -75.5567474, 56.6750221, -75.5148773, 56.6302719, -132.1870117, 132.1898956
21: -98.2285004, 59.6136627, -98.2010040, 59.5244598, -157.7529602, 157.8146667
22: -100.1853714, 61.8789902, -100.1391525, 61.8020782, -161.9874420, 162.0181427
23: -78.3585281, 66.1325760, -78.3387146, 65.9893112, -144.3478241, 144.4712830
24: -96.7707520, 67.2337799, -96.7281418, 67.0820923, -163.8528442, 163.9619141
25: -83.8438721, 68.3056030, -83.8101044, 68.2153931, -152.0592651, 152.1157074
26: -114.8394394, 88.1742325, -114.8016815, 88.0508194, -202.8902588, 202.9759064
27: -98.5043793, 68.6602783, -98.4583511, 68.5675659, -167.0719452, 167.1186218
28: -77.7236481, 67.4547424, -77.7033920, 67.3534393, -145.0770874, 145.1581421
29: -103.9398499, 58.5443230, -103.8876343, 58.4267044, -162.3665466, 162.4319305
30: -98.2107391, 75.3889160, -98.1661835, 75.2354813, -173.4462280, 173.5550842
31: -103.0443726, 70.9969330, -103.0085449, 70.8318634, -173.8762207, 174.0054626
32: -108.8983688, 69.1868439, -108.8624115, 69.1155167, -178.0138855, 178.0492554
33: -135.0750427, 93.2828522, -135.0323792, 93.2308426, -228.3058777, 228.3152161
34: -111.5204926, 68.9299011, -111.4836655, 68.7864227, -180.3069153, 180.4135742
35: -111.3596268, 74.5749207, -111.3260193, 74.4560928, -185.8156891, 185.9009399
36: -114.8042679, 73.8645935, -114.7808456, 73.7918854, -188.5961456, 188.6454468
37: -159.7202301, 73.3709946, -159.6724548, 73.2371292, -232.9573212, 233.0434570
38: -135.1644440, 86.5663681, -135.1285400, 86.5074768, -221.6718903, 221.6949158
39: -151.2538452, 90.2862625, -151.1856232, 90.2716522, -241.5254517, 241.4718781
40: -124.4085388, 69.2478027, -124.3562241, 69.1881409, -193.5966797, 193.6040192
41: -108.4705124, 79.4273987, -108.4464798, 79.3336639, -187.8041687, 187.8738708
42: -79.3239288, 69.0043488, -79.2949448, 68.9637451, -148.2876740, 148.2992859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 723

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2240394, upper bound: 124.3571304
time: 146.57 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.3571304, upper bound: 124.3571304
time: 207.94 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -127.0734787, 89.1233521, -127.3460846, 89.2525864, -216.3260345, 216.4694366
1: -66.4920731, 67.9733582, -66.6468582, 68.1047058, -134.5967712, 134.6202087
2: -58.5326500, 69.1145859, -58.6931877, 69.2686310, -127.8012848, 127.8077698
3: -67.6948395, 78.6202698, -67.8846512, 78.8214874, -146.5162964, 146.5049133
4: -75.2332306, 81.4057465, -75.4840851, 81.6639938, -156.8972168, 156.8898163
5: -65.4728088, 76.0339508, -65.5923309, 76.2062073, -141.6790161, 141.6262817
6: -109.6821823, 78.5604553, -109.8507843, 78.7030334, -188.3852234, 188.4112396
7: -78.7999191, 72.6581268, -78.9882660, 72.8818054, -151.6817169, 151.6463928
8: -87.1319962, 100.3693085, -87.5084839, 100.7829742, -187.9149780, 187.8777924
9: -75.7090302, 75.5121155, -75.8472366, 75.6656799, -151.3747101, 151.3593445
10: -107.3567886, 101.2140045, -107.4892502, 101.3548355, -208.7115784, 208.7032471
11: -103.1987305, 64.2425842, -103.4828415, 64.4025421, -167.6012726, 167.7254333
12: -101.8729324, 80.4162064, -102.0716553, 80.6703568, -182.5432739, 182.4878540
13: -110.8262939, 107.7939148, -111.0104294, 108.0555954, -218.8818970, 218.8043060
14: -160.5100708, 91.0723267, -160.7269287, 91.2327576, -251.7428284, 251.7992554
15: -88.4732742, 76.4162445, -88.6489105, 76.5355377, -165.0087891, 165.0651550
16: -109.8235016, 78.4921341, -110.0595322, 78.6875229, -188.5110168, 188.5516357
17: -156.2149658, 83.9659042, -156.5947266, 84.2277145, -240.4426727, 240.5606384
18: -104.7499847, 77.2910690, -105.1236801, 77.6083374, -182.3583221, 182.4147186
19: -77.9209595, 51.4852257, -78.1185684, 51.5698776, -129.4908295, 129.6037903
20: -75.4902954, 56.6127663, -75.6313858, 56.6976662, -132.1879578, 132.2441559
21: -98.1087189, 59.5111580, -98.3153992, 59.6139908, -157.7227020, 157.8265533
22: -100.0668411, 61.7955933, -100.2909851, 61.9011612, -161.9679871, 162.0865784
23: -78.2297363, 65.9741592, -78.4590378, 66.1554108, -144.3851318, 144.4331970
24: -96.6117249, 67.0742645, -96.8907471, 67.2632446, -163.8749695, 163.9650116
25: -83.7351913, 68.2064438, -83.9212189, 68.3182297, -152.0534210, 152.1276550
26: -114.7287674, 88.0296631, -114.9157410, 88.1997833, -202.9285431, 202.9454041
27: -98.3874283, 68.5555267, -98.5946121, 68.6638260, -167.0512390, 167.1501465
28: -77.6301422, 67.3420258, -77.7946472, 67.4594498, -145.0895844, 145.1366730
29: -103.7922516, 58.4202652, -104.0892410, 58.5869522, -162.3792114, 162.5095062
30: -98.0702820, 75.2165833, -98.3010864, 75.4212952, -173.4915771, 173.5176544
31: -102.8680573, 70.8223114, -103.1763458, 71.0220184, -173.8900452, 173.9986572
32: -108.8271637, 69.0857697, -108.9480591, 69.2265472, -178.0537109, 178.0338135
33: -134.9869385, 93.2117004, -135.1248169, 93.3072662, -228.2941895, 228.3365173
34: -111.3756943, 68.7674942, -111.6396942, 68.9766464, -180.3523407, 180.4071960
35: -111.2282715, 74.4379349, -111.4795914, 74.6304550, -185.8587189, 185.9175110
36: -114.7265778, 73.7817535, -114.8818665, 73.8928757, -188.6194458, 188.6636047
37: -159.5566101, 73.2248459, -159.8599091, 73.4169464, -232.9735565, 233.0847473
38: -135.0851440, 86.4801178, -135.2252808, 86.6066437, -221.6917725, 221.7053833
39: -151.1491089, 90.2098923, -151.3263855, 90.3800888, -241.5292053, 241.5362549
40: -124.3137817, 69.1675568, -124.4824219, 69.2741165, -193.5878601, 193.6499786
41: -108.3826752, 79.3147430, -108.5523911, 79.4556122, -187.8382874, 187.8671265
42: -79.2742310, 68.9386749, -79.3619232, 69.0493622, -148.3235931, 148.3005829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 723

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2237866, upper bound: 124.1890038
time: 159.22 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2237866, upper bound: 124.1890038
time: 118.10 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -127.2407455, 89.1845703, -127.3522339, 89.2547760, -216.4955139, 216.5368042
1: -66.6140671, 68.0520248, -66.6492004, 68.1120911, -134.7261658, 134.7012024
2: -58.6617470, 69.2002640, -58.6947899, 69.2756729, -127.9374237, 127.8950500
3: -67.8363571, 78.7287750, -67.8859177, 78.8316956, -146.6680298, 146.6146851
4: -75.4280472, 81.5427856, -75.4858627, 81.6782455, -157.1062927, 157.0286255
5: -65.5760345, 76.1327362, -65.5940552, 76.2159271, -141.7919464, 141.7267761
6: -109.7745285, 78.6779022, -109.8577194, 78.7051773, -188.4796753, 188.5356140
7: -78.9317245, 72.7721863, -78.9903030, 72.8933105, -151.8250427, 151.7624817
8: -87.4013519, 100.5852280, -87.5099487, 100.8062897, -188.2076263, 188.0951843
9: -75.8284836, 75.6011658, -75.8489685, 75.6737366, -151.5022278, 151.4501343
10: -107.4586868, 101.2938385, -107.4918518, 101.3581467, -208.8168335, 208.7856903
11: -103.3553696, 64.4015198, -103.4974899, 64.4041748, -167.7595520, 167.8990173
12: -101.9713821, 80.5776215, -102.0807343, 80.6738968, -182.6452637, 182.6583405
13: -111.0026932, 107.9570007, -111.0127563, 108.0699692, -219.0726318, 218.9697266
14: -160.6465149, 91.1578064, -160.7316895, 91.2351837, -251.8816986, 251.8894958
15: -88.6188049, 76.4887466, -88.6514587, 76.5417023, -165.1605072, 165.1401978
16: -109.9423828, 78.6133270, -110.0678711, 78.6888580, -188.6312103, 188.6811981
17: -156.3955078, 84.1052170, -156.6074829, 84.2299042, -240.6253967, 240.7127075
18: -104.9601822, 77.5458984, -105.1434708, 77.6097565, -182.5699463, 182.6893616
19: -78.0298309, 51.5655479, -78.1274338, 51.5708389, -129.6006775, 129.6929779
20: -75.5567474, 56.6750221, -75.6341934, 56.6997833, -132.2565308, 132.3092041
21: -98.2285004, 59.6136627, -98.3249969, 59.6155434, -157.8440399, 157.9386597
22: -100.1853714, 61.8789902, -100.2991486, 61.9019165, -162.0872803, 162.1781311
23: -78.3585281, 66.1325760, -78.4715805, 66.1571808, -144.5157166, 144.6041565
24: -96.7707520, 67.2337799, -96.9046478, 67.2640839, -164.0348206, 164.1384125
25: -83.8438721, 68.3056030, -83.9300537, 68.3192444, -152.1631165, 152.2356567
26: -114.8394394, 88.1742325, -114.9238586, 88.2023010, -203.0417480, 203.0980835
27: -98.5043793, 68.6602783, -98.6023560, 68.6652679, -167.1696472, 167.2626343
28: -77.7236481, 67.4547424, -77.8028717, 67.4607697, -145.1844025, 145.2576141
29: -103.9398499, 58.5443230, -104.0992126, 58.5876999, -162.5275421, 162.6435242
30: -98.2107391, 75.3889160, -98.3119278, 75.4234619, -173.6342010, 173.7008362
31: -103.0443726, 70.9969330, -103.1927032, 71.0231094, -174.0674744, 174.1896362
32: -108.8983688, 69.1868439, -108.9518356, 69.2300415, -178.1284180, 178.1386719
33: -135.0750427, 93.2828522, -135.1299133, 93.3093185, -228.3843536, 228.4127655
34: -111.5204926, 68.9299011, -111.6523743, 68.9785767, -180.4990692, 180.5822754
35: -111.3596268, 74.5749207, -111.4911118, 74.6324081, -185.9920044, 186.0660400
36: -114.8042679, 73.8645935, -114.8882599, 73.8939056, -188.6981659, 188.7528534
37: -159.7202301, 73.3709946, -159.8722382, 73.4182358, -233.1384583, 233.2432251
38: -135.1644440, 86.5663681, -135.2301636, 86.6096191, -221.7740479, 221.7965240
39: -151.2538452, 90.2862625, -151.3305969, 90.3869476, -241.6407776, 241.6168518
40: -124.4085388, 69.2478027, -124.4870224, 69.2763290, -193.6848755, 193.7348022
41: -108.4705124, 79.4273987, -108.5595169, 79.4577713, -187.9282684, 187.9869080
42: -79.3239288, 69.0043488, -79.3641891, 69.0519104, -148.3758240, 148.3685303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 610

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 723

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2237841, upper bound: 124.2833685
time: 292.69 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2237841, upper bound: 124.2833685
time: 117.10 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -127.2499771, 88.9006195, -127.1376266, 89.0973206, -216.3472748, 216.0382385
1: -66.5968018, 67.8773041, -66.5102844, 67.9967804, -134.5935822, 134.3875885
2: -58.6530571, 69.0526733, -58.5430145, 69.1528778, -127.8059235, 127.5956879
3: -67.8415527, 78.6310120, -67.7023468, 78.6806488, -146.5222015, 146.3333588
4: -75.4261932, 81.4623337, -75.2428207, 81.4993744, -156.9255524, 156.7051544
5: -65.5497894, 75.9842758, -65.4841461, 76.0750275, -141.6248169, 141.4684143
6: -109.6874313, 78.6195984, -109.7175980, 78.5690460, -188.2564697, 188.3371887
7: -78.9291916, 72.6846848, -78.8161469, 72.7389832, -151.6681671, 151.5008240
8: -87.4466095, 100.4766083, -87.1384125, 100.5293579, -187.9759674, 187.6150208
9: -75.7872238, 75.4368057, -75.7153625, 75.5357513, -151.3229675, 151.1521606
10: -107.3987656, 101.1696625, -107.3700333, 101.2208786, -208.6196136, 208.5396881
11: -103.2282867, 64.3427582, -103.2888336, 64.2490540, -167.4773407, 167.6315918
12: -101.9678345, 80.5522308, -101.9539642, 80.4338531, -182.4016876, 182.5061951
13: -110.9207001, 107.8517990, -110.8365707, 107.8514709, -218.7721710, 218.6883698
14: -160.6166992, 90.9636230, -160.5485840, 91.0605698, -251.6772461, 251.5122070
15: -88.5775452, 76.3531494, -88.4896240, 76.4294662, -165.0069885, 164.8427734
16: -109.9298325, 78.5101700, -109.9034424, 78.4797363, -188.4095612, 188.4136047
17: -156.4640503, 83.9972000, -156.3754578, 83.9589615, -240.4229736, 240.3726501
18: -104.9501572, 77.5506363, -104.8855515, 77.2968674, -182.2470245, 182.4361877
19: -77.8793030, 51.5359268, -77.9668121, 51.4911537, -129.3704529, 129.5027313
20: -75.3858032, 56.6521759, -75.4802933, 56.6363792, -132.0221863, 132.1324768
21: -98.0079498, 59.5669327, -98.1431274, 59.5198212, -157.5277710, 157.7100525
22: -99.9796448, 61.8524971, -100.1106415, 61.7966461, -161.7762909, 161.9631348
23: -78.2504730, 66.1011276, -78.2960358, 65.9822235, -144.2326813, 144.3971558
24: -96.5870438, 67.2233734, -96.6736145, 67.0774002, -163.6644287, 163.8969879
25: -83.6162186, 68.2622375, -83.7608337, 68.2082367, -151.8244629, 152.0230713
26: -114.7325668, 88.1391602, -114.7597427, 88.0463791, -202.7789459, 202.8988953
27: -98.3535538, 68.6175308, -98.4152832, 68.5611572, -166.9147034, 167.0328064
28: -77.5607758, 67.4179916, -77.6566620, 67.3478775, -144.9086609, 145.0746460
29: -103.8106842, 58.5377312, -103.8761902, 58.4194336, -162.2301025, 162.4139099
30: -97.9702225, 75.3579254, -98.1057739, 75.2272034, -173.1974182, 173.4636993
31: -102.8470383, 70.9698868, -102.9435577, 70.8263245, -173.6733551, 173.9134521
32: -108.8054276, 69.1369629, -108.8404846, 69.1042557, -177.9096527, 177.9774475
33: -134.8119965, 93.2366028, -134.9783020, 93.2260284, -228.0380096, 228.2149048
34: -111.4524460, 68.9203644, -111.4620667, 68.7827911, -180.2352295, 180.3824158
35: -111.2553940, 74.5711975, -111.3079910, 74.4514847, -185.7068787, 185.8791809
36: -114.6945877, 73.8381348, -114.7688446, 73.7860413, -188.4806213, 188.6069641
37: -159.6135559, 73.3575134, -159.6468811, 73.2307892, -232.8443298, 233.0043945
38: -135.0142212, 86.5143661, -135.0956268, 86.5016327, -221.5158539, 221.6099854
39: -151.0528870, 90.3105316, -151.1506958, 90.2852859, -241.3381653, 241.4612122
40: -124.3242188, 69.1349564, -124.3342590, 69.1757965, -193.5000153, 193.4692078
41: -108.4582214, 79.3879318, -108.4359283, 79.3239288, -187.7821350, 187.8238525
42: -79.2822418, 68.9677734, -79.2838669, 68.9689026, -148.2511444, 148.2516327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=750, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 721

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.1449587, upper bound: 124.2548340
time: 106.94 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.1449587, upper bound: 124.2548340
time: 133.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 243.06 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.2240394, upper bound: 124.2541357
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.3571304, upper bound: 124.2541357
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.2240394, upper bound: 124.3571304
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.3571304, upper bound: 124.3571304
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.2237866, upper bound: 124.1890038
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.2237866, upper bound: 124.1890038
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.2237841, upper bound: 124.2833685
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.2237841, upper bound: 124.2833685
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.1449587, upper bound: 124.2548340
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 243.06
Output dim: 8, lower bound: -124.1449587, upper bound: 124.2548340
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 243.06
Output dim: 8, lower bound: -124.2973633, upper bound: 124.3929445
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 243.06
Output dim: 8, lower bound: -124.2973633, upper bound: 124.2604688
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 243.06
Output dim: 8, lower bound: -124.2973633, upper bound: 124.3929445
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=187.93016052246094
rel_dist={8: [-124.44835295992456, 124.44835294926614]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1785

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7472179, upper bound: 121.6533666
time: 98.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7472179, upper bound: 121.7472177
time: 222.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 320.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 320.89
Output dim: 8, lower bound: -121.7472179, upper bound: 121.6533666
IS_A2, status: Status.UNKNOWN, split count: 1, time: 320.89
Output dim: 8, lower bound: -121.7472179, upper bound: 121.7472177

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -127.1311264, 89.1436310, -127.1961441, 89.1595383, -216.2906647, 216.3397827
1: -66.5138092, 68.0384521, -66.5339432, 68.0800781, -134.5938873, 134.5723877
2: -58.5468636, 69.1890488, -58.5595360, 69.2413788, -127.7882385, 127.7485809
3: -67.7071228, 78.7096252, -67.7178345, 78.7774353, -146.4845581, 146.4274597
4: -75.2487946, 81.5308990, -75.2625885, 81.6263885, -156.8751831, 156.7934723
5: -65.4886627, 76.1151276, -65.5023422, 76.1710052, -141.6596680, 141.6174622
6: -109.7483292, 78.5794678, -109.7938385, 78.5951385, -188.3434448, 188.3733063
7: -78.8203201, 72.7568512, -78.8399353, 72.8412018, -151.6615295, 151.5967712
8: -87.1449585, 100.5661545, -87.1567841, 100.7252655, -187.8701935, 187.7229309
9: -75.7243347, 75.5817566, -75.7358093, 75.6243286, -151.3486633, 151.3175659
10: -107.3807678, 101.2442780, -107.4011230, 101.2688370, -208.6495819, 208.6454010
11: -103.3352585, 64.2569122, -103.4352417, 64.2684784, -167.6037292, 167.6921539
12: -101.9506149, 80.4471130, -102.0260544, 80.4735718, -182.4241791, 182.4731598
13: -110.8471680, 107.9141769, -110.8650513, 107.9819794, -218.8291473, 218.7792358
14: -160.5510559, 91.0945892, -160.5963440, 91.1125946, -251.6636505, 251.6909027
15: -88.4957047, 76.4759598, -88.5163727, 76.5059586, -165.0016632, 164.9923401
16: -109.9002151, 78.5042496, -109.9770966, 78.5133057, -188.4135132, 188.4813538
17: -156.3343353, 83.9851913, -156.4768982, 84.0039215, -240.3382568, 240.4620819
18: -104.9205704, 77.3037720, -105.0468063, 77.3146210, -182.2351990, 182.3505859
19: -78.0098114, 51.4939690, -78.0698547, 51.5024109, -129.5122223, 129.5638123
20: -75.5152817, 56.6305389, -75.5327759, 56.6574059, -132.1726685, 132.1632996
21: -98.2035141, 59.5246239, -98.2624130, 59.5366135, -157.7401123, 157.7870331
22: -100.1405792, 61.8021851, -100.2073593, 61.8083725, -161.9489441, 162.0095520
23: -78.3402786, 65.9895554, -78.4149933, 66.0017700, -144.3420410, 144.4045258
24: -96.7298279, 67.0822754, -96.8107147, 67.0891571, -163.8189850, 163.8929749
25: -83.8112106, 68.2155304, -83.8632355, 68.2231522, -152.0343628, 152.0787659
26: -114.8033524, 88.0512085, -114.8473587, 88.0710754, -202.8744049, 202.8985596
27: -98.4599075, 68.5677719, -98.5070572, 68.5771637, -167.0370636, 167.0748138
28: -77.7048492, 67.3536377, -77.7505341, 67.3628082, -145.0676575, 145.1041565
29: -103.8900681, 58.4268112, -103.9851685, 58.4314613, -162.3215332, 162.4119873
30: -98.1680603, 75.2357712, -98.2311249, 75.2510071, -173.4190674, 173.4668884
31: -103.0106888, 70.8320160, -103.1049194, 70.8409882, -173.8516541, 173.9369354
32: -108.8633575, 69.1159210, -108.8888474, 69.1400757, -178.0034180, 178.0047607
33: -135.0333557, 93.2313156, -135.0592194, 93.2507935, -228.2841492, 228.2905273
34: -111.4851456, 68.7869720, -111.5733337, 68.8054276, -180.2905731, 180.3603058
35: -111.3274078, 74.4565659, -111.4141693, 74.4739075, -185.8013153, 185.8707275
36: -114.7816772, 73.7921448, -114.8346558, 73.8015060, -188.5831757, 188.6268005
37: -159.6750336, 73.2374573, -159.7719421, 73.2487106, -232.9237366, 233.0093994
38: -135.1294861, 86.5081635, -135.1599426, 86.5361557, -221.6656494, 221.6680603
39: -151.1862946, 90.2730103, -151.2161560, 90.3408356, -241.5271301, 241.4891663
40: -124.3573074, 69.1886826, -124.3903351, 69.2109985, -193.5682983, 193.5790100
41: -108.4479294, 79.3339767, -108.5003662, 79.3484344, -187.7963562, 187.8343506
42: -79.2953491, 68.9645996, -79.3113708, 68.9992218, -148.2945557, 148.2759705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=524, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6509083, upper bound: 121.6386800
time: 139.21 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6509083, upper bound: 121.6386800
time: 136.59 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -127.3538971, 89.2551880, -127.2050171, 89.1627960, -216.5166931, 216.4602051
1: -66.6497192, 68.1133728, -66.5382538, 68.0891190, -134.7388306, 134.6516266
2: -58.6951294, 69.2781219, -58.5623016, 69.2528915, -127.9480209, 127.8404236
3: -67.8862610, 78.8332672, -67.7200012, 78.7937012, -146.6799469, 146.5532684
4: -75.4861374, 81.6803513, -75.2655487, 81.6487274, -157.1348572, 156.9458923
5: -65.5944061, 76.2170715, -65.5051804, 76.1838226, -141.7782288, 141.7222443
6: -109.8595734, 78.7055054, -109.8039246, 78.5987320, -188.4583130, 188.5094299
7: -78.9908905, 72.8946762, -78.8436050, 72.8619385, -151.8528290, 151.7382812
8: -87.5101776, 100.8090134, -87.1591339, 100.7636871, -188.2738647, 187.9681396
9: -75.8492584, 75.6747284, -75.7382660, 75.6342010, -151.4834595, 151.4129944
10: -107.4923935, 101.3588181, -107.4054031, 101.2745361, -208.7669220, 208.7642212
11: -103.5003204, 64.4044418, -103.4556351, 64.2708435, -167.7711639, 167.8600769
12: -102.0819778, 80.6744537, -102.0441437, 80.4796371, -182.5615845, 182.7185974
13: -111.0131760, 108.0716629, -110.8690033, 107.9969559, -219.0101166, 218.9406738
14: -160.7323914, 91.2356033, -160.6016083, 91.1167526, -251.8491516, 251.8371887
15: -88.6518555, 76.5433502, -88.5209198, 76.5121918, -165.1640472, 165.0642700
16: -110.0696335, 78.6890869, -109.9949799, 78.5153580, -188.5849915, 188.6840668
17: -156.6102295, 84.2303314, -156.5096436, 84.0081635, -240.6183777, 240.7399597
18: -105.1457748, 77.6100159, -105.0771637, 77.3170319, -182.4627991, 182.6871643
19: -78.1299133, 51.5710678, -78.0837555, 51.5043411, -129.6342468, 129.6548157
20: -75.6346283, 56.7000542, -75.5365448, 56.6574593, -132.2920837, 132.2366028
21: -98.3277283, 59.6157417, -98.2752304, 59.5393372, -157.8670654, 157.8909760
22: -100.3006821, 61.9020538, -100.2234650, 61.8093643, -162.1100464, 162.1255035
23: -78.4733353, 66.1574326, -78.4322357, 66.0044708, -144.4777985, 144.5896606
24: -96.9062805, 67.2642975, -96.8304672, 67.0905914, -163.9968719, 164.0947571
25: -83.9311523, 68.3194427, -83.8755798, 68.2243958, -152.1555481, 152.1950073
26: -114.9255447, 88.2026978, -114.8571701, 88.0756989, -203.0012207, 203.0598450
27: -98.6040497, 68.6654434, -98.5177460, 68.5793076, -167.1833496, 167.1831970
28: -77.8043823, 67.4609833, -77.7603607, 67.3647614, -145.1691437, 145.2213440
29: -104.1020508, 58.5877991, -104.0088577, 58.4322128, -162.5342712, 162.5966492
30: -98.3139038, 75.4237823, -98.2449493, 75.2542572, -173.5681610, 173.6687317
31: -103.1951599, 71.0233078, -103.1269989, 70.8429260, -174.0380859, 174.1502991
32: -108.9527740, 69.2304688, -108.8946457, 69.1457825, -178.0985413, 178.1250916
33: -135.1309052, 93.3098450, -135.0650940, 93.2540970, -228.3849945, 228.3749390
34: -111.6538925, 68.9791489, -111.5945435, 68.8091965, -180.4630890, 180.5737000
35: -111.4924774, 74.6328888, -111.4350739, 74.4773865, -185.9698486, 186.0679626
36: -114.8892288, 73.8942490, -114.8473129, 73.8033829, -188.6926117, 188.7415466
37: -159.8751068, 73.4185944, -159.7944031, 73.2507553, -233.1258545, 233.2129974
38: -135.2311096, 86.6103363, -135.1668243, 86.5405807, -221.7716675, 221.7771606
39: -151.3312378, 90.3884811, -151.2229309, 90.3562775, -241.6875153, 241.6114044
40: -124.4881058, 69.2769165, -124.3978577, 69.2143707, -193.7024536, 193.6747742
41: -108.5610733, 79.4581070, -108.5118713, 79.3517609, -187.9128418, 187.9699554
42: -79.3646469, 69.0527725, -79.3147278, 69.0035400, -148.3681793, 148.3674774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=524, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6509083, upper bound: 121.7350951
time: 209.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6509083, upper bound: 121.7350951
time: 129.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 341.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 341.48
Output dim: 8, lower bound: -121.6509083, upper bound: 121.6386800
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 341.48
Output dim: 8, lower bound: -121.6509083, upper bound: 121.6386800
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 341.48
Output dim: 8, lower bound: -121.6509083, upper bound: 121.7350951
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 341.48
Output dim: 8, lower bound: -121.6509083, upper bound: 121.7350951

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -127.1159897, 89.1384583, -127.1381683, 89.1395264, -216.2554932, 216.2766266
1: -66.5081253, 68.0214081, -66.5121994, 68.0157852, -134.5239105, 134.5335999
2: -58.5431290, 69.1701431, -58.5453186, 69.1679993, -127.7111206, 127.7154617
3: -67.7039185, 78.6864166, -67.7055969, 78.6879349, -146.3918457, 146.3920135
4: -75.2447662, 81.4984207, -75.2471008, 81.5012054, -156.7459564, 156.7455139
5: -65.4845428, 76.0938950, -65.4865494, 76.0897675, -141.5743103, 141.5804443
6: -109.7311249, 78.5745392, -109.7277298, 78.5761490, -188.3072510, 188.3022766
7: -78.8150177, 72.7316284, -78.8195648, 72.7424316, -151.5574493, 151.5511932
8: -87.1416016, 100.5151978, -87.1438904, 100.5282135, -187.6698151, 187.6590729
9: -75.7203445, 75.5640182, -75.7204895, 75.5546799, -151.2750244, 151.2845001
10: -107.3745499, 101.2365341, -107.3771362, 101.2387314, -208.6132812, 208.6136780
11: -103.3002167, 64.2532043, -103.2986221, 64.2542267, -167.5544434, 167.5518188
12: -101.9303513, 80.4391174, -101.9483261, 80.4427948, -182.3731384, 182.3874207
13: -110.8417816, 107.8828049, -110.8442001, 107.8616486, -218.7034302, 218.7270050
14: -160.5404510, 91.0889359, -160.5552979, 91.0905762, -251.6310272, 251.6442261
15: -88.4899292, 76.4606476, -88.4939880, 76.4462433, -164.9361725, 164.9546356
16: -109.8802185, 78.5011444, -109.9010696, 78.5012207, -188.3814392, 188.4021912
17: -156.3036041, 83.9801941, -156.3574829, 83.9847107, -240.2883148, 240.3376465
18: -104.8772888, 77.3005066, -104.8760757, 77.3019257, -182.1791992, 182.1765594
19: -77.9871063, 51.4917221, -77.9827271, 51.4936371, -129.4807281, 129.4744263
20: -75.5088654, 56.6259270, -75.5078506, 56.6395988, -132.1484528, 132.1337738
21: -98.1790314, 59.5211678, -98.1686020, 59.5231552, -157.7021790, 157.6897736
22: -100.1213684, 61.8004646, -100.1338959, 61.8017731, -161.9231415, 161.9343567
23: -78.3119354, 65.9855957, -78.3043823, 65.9864120, -144.2983398, 144.2899780
24: -96.6995163, 67.0801849, -96.6924133, 67.0811920, -163.7807007, 163.7725830
25: -83.7917786, 68.2132034, -83.7871399, 68.2141113, -152.0058594, 152.0003357
26: -114.7841873, 88.0455322, -114.7727814, 88.0494537, -202.8336029, 202.8182983
27: -98.4412231, 68.5645905, -98.4348907, 68.5649109, -167.0061340, 166.9994812
28: -77.6857300, 67.3506012, -77.6759186, 67.3512268, -145.0369568, 145.0265198
29: -103.8644867, 58.4251099, -103.8873749, 58.4249420, -162.2893982, 162.3124847
30: -98.1429367, 75.2308350, -98.1332321, 75.2319336, -173.3748779, 173.3640747
31: -102.9734573, 70.8295212, -102.9622955, 70.8312454, -173.8047028, 173.7918091
32: -108.8540497, 69.1082611, -108.8527374, 69.1099625, -177.9640198, 177.9609985
33: -135.0214233, 93.2262421, -135.0128784, 93.2311401, -228.2525635, 228.2391052
34: -111.4572983, 68.7819519, -111.4638214, 68.7859802, -180.2432861, 180.2457733
35: -111.3021545, 74.4517670, -111.3149338, 74.4553070, -185.7574615, 185.7666931
36: -114.7672424, 73.7894592, -114.7795563, 73.7911835, -188.5584106, 188.5690002
37: -159.6447754, 73.2342072, -159.6539612, 73.2361374, -232.8809204, 232.8881531
38: -135.1180267, 86.5009308, -135.1156616, 86.5080948, -221.6261139, 221.6165924
39: -151.1766815, 90.2563629, -151.1791077, 90.2776794, -241.4543610, 241.4354706
40: -124.3461685, 69.1831970, -124.3468018, 69.1898499, -193.5360107, 193.5299988
41: -108.4309998, 79.3290634, -108.4352722, 79.3292542, -187.7602234, 187.7643433
42: -79.2899323, 68.9579620, -79.2904510, 68.9732819, -148.2631989, 148.2484131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6456597, upper bound: 121.5186263
time: 128.69 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6336356
time: 178.25 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -127.1281052, 89.1429138, -127.3061676, 89.2006989, -216.3287964, 216.4490814
1: -66.5129242, 68.0365295, -66.6342773, 68.0935211, -134.6064453, 134.6707916
2: -58.5462608, 69.1855469, -58.6745224, 69.2523651, -127.7986145, 127.8600616
3: -67.7066116, 78.7074127, -67.8470001, 78.7965088, -146.5031128, 146.5544128
4: -75.2483215, 81.5278091, -75.4418030, 81.6381149, -156.8864441, 156.9696045
5: -65.4880524, 76.1131897, -65.5896759, 76.1884766, -141.6765289, 141.7028503
6: -109.7451324, 78.5789108, -109.8200378, 78.6936188, -188.4387512, 188.3989563
7: -78.8193665, 72.7545013, -78.9512711, 72.8564148, -151.6757812, 151.7057648
8: -87.1446075, 100.5614166, -87.4131012, 100.7442780, -187.8888855, 187.9745178
9: -75.7238922, 75.5800934, -75.8399048, 75.6436920, -151.3675842, 151.4199982
10: -107.3798904, 101.2432632, -107.4789505, 101.3185577, -208.6984406, 208.7222137
11: -103.3311996, 64.2564850, -103.4552002, 64.4130936, -167.7442932, 167.7116852
12: -101.9485245, 80.4462280, -102.0467529, 80.6040115, -182.5525208, 182.4929657
13: -110.8464508, 107.9112396, -111.0205154, 108.0248642, -218.8713074, 218.9317627
14: -160.5498047, 91.0939484, -160.6918793, 91.1759033, -251.7257080, 251.7858276
15: -88.4950867, 76.4731293, -88.6394196, 76.5187225, -165.0138092, 165.1125488
16: -109.8975372, 78.5038528, -110.0186310, 78.6223907, -188.5199280, 188.5224609
17: -156.3300934, 83.9845276, -156.5378876, 84.1242676, -240.4543610, 240.5223999
18: -104.9165802, 77.3033752, -105.0862885, 77.5567017, -182.4732819, 182.3896637
19: -78.0060577, 51.4936218, -78.0896912, 51.5740128, -129.5800781, 129.5833130
20: -75.5145569, 56.6300697, -75.5742188, 56.7019424, -132.2164917, 132.2042847
21: -98.1991882, 59.5243340, -98.2870712, 59.6256027, -157.8247986, 157.8114014
22: -100.1381226, 61.8020020, -100.2518311, 61.8851242, -162.0232544, 162.0538330
23: -78.3375549, 65.9891281, -78.4332428, 66.1447678, -144.4823303, 144.4223633
24: -96.7268982, 67.0819626, -96.8520432, 67.2406006, -163.9674988, 163.9340057
25: -83.8093262, 68.2152786, -83.8958588, 68.3131943, -152.1225281, 152.1111298
26: -114.8004379, 88.0505447, -114.8834381, 88.1942520, -202.9946899, 202.9339905
27: -98.4572296, 68.5674515, -98.5515823, 68.6696548, -167.1268921, 167.1190338
28: -77.7023468, 67.3532715, -77.7692566, 67.4639511, -145.1662903, 145.1225281
29: -103.8861847, 58.4266243, -104.0343781, 58.5489235, -162.4351044, 162.4609985
30: -98.1651230, 75.2352676, -98.2738037, 75.4040985, -173.5692139, 173.5090485
31: -103.0069427, 70.8317261, -103.1386337, 71.0058289, -174.0127716, 173.9703674
32: -108.8617859, 69.1152191, -108.9240265, 69.2109985, -178.0727844, 178.0392456
33: -135.0316467, 93.2304535, -135.1010284, 93.3020935, -228.3337402, 228.3314667
34: -111.4825668, 68.7859802, -111.6086502, 68.9483490, -180.4309082, 180.3946228
35: -111.3250351, 74.4557800, -111.4464264, 74.5922699, -185.9172974, 185.9022064
36: -114.7803345, 73.7916718, -114.8570938, 73.8739624, -188.6542664, 188.6487732
37: -159.6709595, 73.2369232, -159.8171692, 73.3820877, -233.0530396, 233.0540771
38: -135.1278687, 86.5069580, -135.1949921, 86.5934906, -221.7213593, 221.7019501
39: -151.1851196, 90.2707748, -151.2837982, 90.3539429, -241.5390625, 241.5545654
40: -124.3554230, 69.1877136, -124.4416580, 69.2695847, -193.6250000, 193.6293640
41: -108.4454346, 79.3333893, -108.5230331, 79.4418411, -187.8872681, 187.8564148
42: -79.2946777, 68.9631042, -79.3397675, 69.0388184, -148.3334961, 148.3028717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7296278, upper bound: 121.5186263
time: 298.75 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6336356
time: 126.09 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -127.3386230, 89.2500305, -127.1466904, 89.1428528, -216.4814758, 216.3967285
1: -66.6440887, 68.0966415, -66.5164490, 68.0250244, -134.6690979, 134.6130829
2: -58.6913834, 69.2595215, -58.5480042, 69.1795578, -127.8709412, 127.8075256
3: -67.8831177, 78.8105774, -67.7078018, 78.7042542, -146.5873718, 146.5183716
4: -75.4821472, 81.6483765, -75.2500458, 81.5235138, -157.0056610, 156.8984222
5: -65.5903320, 76.1960907, -65.4893875, 76.1026001, -141.6929169, 141.6854858
6: -109.8423691, 78.7006226, -109.7377167, 78.5797272, -188.4220886, 188.4383392
7: -78.9856567, 72.8694763, -78.8232574, 72.7631378, -151.7487793, 151.6927338
8: -87.5068665, 100.7580032, -87.1462784, 100.5665131, -188.0733795, 187.9042816
9: -75.8453217, 75.6570129, -75.7230072, 75.5645447, -151.4098663, 151.3800201
10: -107.4862442, 101.3511200, -107.3814468, 101.2444839, -208.7307281, 208.7325745
11: -103.4662628, 64.4007797, -103.3189774, 64.2565918, -167.7228546, 167.7197571
12: -102.0618362, 80.6665039, -101.9663620, 80.4488525, -182.5106812, 182.6328735
13: -111.0078506, 108.0405045, -110.8481522, 107.8764954, -218.8843384, 218.8886414
14: -160.7217407, 91.2300110, -160.5603943, 91.0948257, -251.8165588, 251.7904053
15: -88.6461182, 76.5280380, -88.4985275, 76.4523926, -165.0985107, 165.0265656
16: -110.0498734, 78.6860275, -109.9188995, 78.5032883, -188.5531616, 188.6049194
17: -156.5798035, 84.2252197, -156.3902130, 83.9889221, -240.5687256, 240.6154327
18: -105.1024780, 77.6067581, -104.9063492, 77.3043518, -182.4067993, 182.5131073
19: -78.1077118, 51.5688019, -77.9966583, 51.4955521, -129.6032715, 129.5654602
20: -75.6282730, 56.6953888, -75.5116806, 56.6395798, -132.2678528, 132.2070618
21: -98.3036041, 59.6123047, -98.1814575, 59.5258675, -157.8294525, 157.7937469
22: -100.2816849, 61.9003601, -100.1499786, 61.8027916, -162.0844727, 162.0503235
23: -78.4453354, 66.1534882, -78.3215637, 65.9890976, -144.4344330, 144.4750519
24: -96.8758850, 67.2622528, -96.7118683, 67.0826721, -163.9585571, 163.9741211
25: -83.9117432, 68.3170929, -83.7994537, 68.2153778, -152.1271210, 152.1165466
26: -114.9064178, 88.1970062, -114.7826538, 88.0539551, -202.9603577, 202.9796448
27: -98.5855637, 68.6623001, -98.4455566, 68.5670776, -167.1526489, 167.1078491
28: -77.7854004, 67.4579620, -77.6858292, 67.3531494, -145.1385498, 145.1437988
29: -104.0770721, 58.5861130, -103.9109497, 58.4257126, -162.5027771, 162.4970703
30: -98.2889252, 75.4189148, -98.1472321, 75.2351761, -173.5241089, 173.5661469
31: -103.1583176, 71.0207672, -102.9847488, 70.8331757, -173.9914856, 174.0055237
32: -108.9435043, 69.2227936, -108.8585358, 69.1156769, -178.0591583, 178.0813293
33: -135.1190796, 93.3048096, -135.0187836, 93.2345581, -228.3536377, 228.3235931
34: -111.6261139, 68.9741974, -111.4850311, 68.7897415, -180.4158325, 180.4592285
35: -111.4672394, 74.6281433, -111.3357849, 74.4587936, -185.9260254, 185.9639282
36: -114.8749161, 73.8915634, -114.7923279, 73.7930298, -188.6679382, 188.6838989
37: -159.8453369, 73.4152985, -159.6769714, 73.2382202, -233.0835571, 233.0922699
38: -135.2197876, 86.6031494, -135.1226501, 86.5124359, -221.7322235, 221.7257996
39: -151.3217316, 90.3720703, -151.1858673, 90.2931213, -241.6148376, 241.5579224
40: -124.4770203, 69.2714310, -124.3543091, 69.1931610, -193.6701660, 193.6257324
41: -108.5442352, 79.4531631, -108.4469833, 79.3325500, -187.8767853, 187.9001465
42: -79.3593140, 69.0461273, -79.2938690, 68.9775848, -148.3368988, 148.3399963

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6148778
time: 195.58 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6456597, upper bound: 121.7296275
time: 122.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -127.3510208, 89.2544708, -127.3150711, 89.2039337, -216.5549622, 216.5695190
1: -66.6488190, 68.1111450, -66.6385651, 68.1018524, -134.7506714, 134.7496948
2: -58.6945305, 69.2742233, -58.6772461, 69.2628021, -127.9573288, 127.9514618
3: -67.8856812, 78.8305435, -67.8491440, 78.8127747, -146.6984558, 146.6796875
4: -75.4856415, 81.6767044, -75.4447174, 81.6603165, -157.1459503, 157.1214142
5: -65.5937500, 76.2150879, -65.5924606, 76.2012482, -141.7949982, 141.8075562
6: -109.8563766, 78.7049713, -109.8300247, 78.6972504, -188.5536041, 188.5349731
7: -78.9898834, 72.8923111, -78.9549103, 72.8770599, -151.8669434, 151.8472290
8: -87.5097580, 100.8042603, -87.4154434, 100.7825775, -188.2923279, 188.2196960
9: -75.8487549, 75.6730194, -75.8423309, 75.6535339, -151.5022736, 151.5153198
10: -107.4914780, 101.3577118, -107.4832382, 101.3242111, -208.8156738, 208.8409424
11: -103.4956131, 64.4039612, -103.4754868, 64.4154129, -167.9110260, 167.8794556
12: -102.0798492, 80.6734924, -102.0647583, 80.6100388, -182.6898499, 182.7382507
13: -111.0123978, 108.0687180, -111.0244827, 108.0397110, -219.0520935, 219.0932007
14: -160.7311401, 91.2348938, -160.6970062, 91.1800308, -251.9111481, 251.9318848
15: -88.6511536, 76.5405121, -88.6439667, 76.5249023, -165.1760559, 165.1844635
16: -110.0666351, 78.6886902, -110.0360184, 78.6244507, -188.6910858, 188.7247009
17: -156.6054993, 84.2296753, -156.5705261, 84.1285400, -240.7340393, 240.8001709
18: -105.1417389, 77.6095581, -105.1165848, 77.5590897, -182.7008362, 182.7261353
19: -78.1255798, 51.5707092, -78.1020050, 51.5759354, -129.7015076, 129.6727142
20: -75.6338806, 56.6995888, -75.5779877, 56.7020073, -132.3358917, 132.2775726
21: -98.3229675, 59.6154213, -98.2990799, 59.6283379, -157.9512939, 157.9145050
22: -100.2980194, 61.9018364, -100.2676239, 61.8861237, -162.1841431, 162.1694489
23: -78.4704132, 66.1569901, -78.4504242, 66.1474304, -144.6178284, 144.6074066
24: -96.9034424, 67.2639465, -96.8717957, 67.2420349, -164.1454773, 164.1357422
25: -83.9292526, 68.3191223, -83.9081726, 68.3144150, -152.2436676, 152.2272949
26: -114.9225998, 88.2020340, -114.8932190, 88.1988449, -203.1214294, 203.0952454
27: -98.6011353, 68.6651154, -98.5618896, 68.6718292, -167.2729645, 167.2270050
28: -77.8017960, 67.4606247, -77.7789917, 67.4658585, -145.2676544, 145.2396088
29: -104.0974579, 58.5876045, -104.0576935, 58.5496597, -162.6471252, 162.6452942
30: -98.3107529, 75.4232483, -98.2875137, 75.4073181, -173.7180786, 173.7107544
31: -103.1909256, 71.0229645, -103.1605682, 71.0077896, -174.1987152, 174.1835327
32: -108.9511795, 69.2297211, -108.9297333, 69.2167130, -178.1678925, 178.1594543
33: -135.1291962, 93.3089142, -135.1069641, 93.3053741, -228.4345398, 228.4158783
34: -111.6512451, 68.9781494, -111.6298065, 68.9520874, -180.6033325, 180.6079407
35: -111.4900818, 74.6320572, -111.4672623, 74.5957031, -186.0857697, 186.0993042
36: -114.8876648, 73.8937225, -114.8696213, 73.8757629, -188.7634125, 188.7633362
37: -159.8703308, 73.4179153, -159.8387299, 73.3841095, -233.2544403, 233.2566528
38: -135.2294617, 86.6090851, -135.2019043, 86.5975418, -221.8269958, 221.8109589
39: -151.3300781, 90.3859787, -151.2904663, 90.3693619, -241.6994324, 241.6764374
40: -124.4862289, 69.2758789, -124.4491272, 69.2726898, -193.7589111, 193.7250061
41: -108.5583954, 79.4575119, -108.5341110, 79.4451141, -188.0035095, 187.9916077
42: -79.3638916, 69.0512695, -79.3430481, 69.0431366, -148.4070282, 148.3943176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6148778
time: 148.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6336356
time: 529.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 680.15 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 680.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.5186263
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 680.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6336356
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 680.15
Output dim: 8, lower bound: -121.7296278, upper bound: 121.5186263
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 680.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6336356
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 680.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6148778
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 680.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.7296275
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 680.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6148778
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 680.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6336356

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -127.0199890, 88.7864761, -127.1132431, 89.0489655, -216.0689545, 215.8997192
1: -66.4581146, 67.7939911, -66.4991074, 67.9580841, -134.4161835, 134.2930908
2: -58.5029907, 68.9541702, -58.5348778, 69.1131363, -127.6161194, 127.4890442
3: -67.6608353, 78.4959564, -67.6943512, 78.6392212, -146.3000488, 146.1903076
4: -75.1868515, 81.2967072, -75.2321625, 81.4499969, -156.6368408, 156.5288696
5: -65.4419708, 75.8719482, -65.4754944, 76.0332260, -141.4751892, 141.3474426
6: -109.5677643, 78.4911423, -109.6859970, 78.5545807, -188.1223297, 188.1771393
7: -78.7559357, 72.5344696, -78.8043137, 72.6923676, -151.4483032, 151.3387756
8: -87.0797272, 100.2088318, -87.1278076, 100.4504089, -187.5301361, 187.3366394
9: -75.6603241, 75.3351288, -75.7049408, 75.4964600, -151.1567688, 151.0400696
10: -107.2840729, 101.0513229, -107.3536682, 101.1913223, -208.4754028, 208.4049988
11: -103.0457153, 64.1933899, -103.2334137, 64.2387238, -167.2844238, 167.4268036
12: -101.8265457, 80.3209763, -101.9215622, 80.4124527, -182.2389984, 182.2425385
13: -110.7520294, 107.6790009, -110.8207245, 107.8091736, -218.5611725, 218.4997253
14: -160.4302063, 90.8198395, -160.5266418, 91.0223694, -251.4525604, 251.3464661
15: -88.4185867, 76.2782974, -88.4755707, 76.3994904, -164.8180695, 164.7538757
16: -109.7505875, 78.3238144, -109.8675766, 78.4544830, -188.2050629, 188.1913910
17: -156.1729736, 83.7497177, -156.3233948, 83.9250488, -240.0980225, 240.0731049
18: -104.7037277, 77.2427444, -104.8313675, 77.2868195, -181.9905396, 182.0741119
19: -77.7478027, 51.4577942, -77.9216003, 51.4846725, -129.2324829, 129.3793945
20: -75.2632294, 56.5804367, -75.4452820, 56.6277122, -131.8909302, 132.0257263
21: -97.8715820, 59.4741287, -98.0902557, 59.5107994, -157.3823853, 157.5643616
22: -99.8099594, 61.7518272, -100.0546417, 61.7891808, -161.5991364, 161.8064728
23: -78.1033630, 65.9313049, -78.2511292, 65.9723206, -144.0756836, 144.1824341
24: -96.3957977, 67.0403366, -96.6148224, 67.0705795, -163.4663544, 163.6551514
25: -83.4867935, 68.1571503, -83.7093353, 68.1994934, -151.6862793, 151.8664856
26: -114.6010361, 87.9849777, -114.7259293, 88.0337982, -202.6347961, 202.7108917
27: -98.2001648, 68.5182800, -98.3733597, 68.5529175, -166.7530670, 166.8916321
28: -77.4518433, 67.3091125, -77.6163025, 67.3404236, -144.7922668, 144.9254150
29: -103.5858917, 58.3758888, -103.8163605, 58.4122162, -161.9981079, 162.1922455
30: -97.8120804, 75.1674500, -98.0489197, 75.2154999, -173.0275879, 173.2163696
31: -102.6441040, 70.7773819, -102.8780975, 70.8174515, -173.4615479, 173.6554871
32: -108.7113953, 69.0186234, -108.8161926, 69.0867691, -177.7981262, 177.8348083
33: -134.7085571, 93.1556091, -134.9323578, 93.2129669, -227.9215088, 228.0879669
34: -111.2700500, 68.7256775, -111.4160309, 68.7713699, -180.0414124, 180.1417084
35: -111.0779877, 74.3925018, -111.2577286, 74.4399948, -185.5179749, 185.6502380
36: -114.5799484, 73.7346725, -114.7316971, 73.7768936, -188.3568420, 188.4663696
37: -159.3984375, 73.1747055, -159.5909576, 73.2208099, -232.6192474, 232.7656555
38: -134.9069214, 86.4087219, -135.0614319, 86.4842377, -221.3911591, 221.4701538
39: -150.9032288, 90.1868286, -151.1087799, 90.2597351, -241.1629486, 241.2956085
40: -124.1879501, 69.0439758, -124.3060608, 69.1535187, -193.3414612, 193.3500214
41: -108.3367767, 79.2613983, -108.4111633, 79.3117828, -187.6485443, 187.6725616
42: -79.2102280, 68.8763885, -79.2699738, 68.9522400, -148.1624603, 148.1463623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=750, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 721

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.5164843
time: 157.76 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.5164843
time: 106.24 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -127.2903137, 89.1431274, -127.1323318, 89.1312256, -216.4214935, 216.2754517
1: -66.6336670, 68.0248413, -66.5088120, 68.0108490, -134.6445160, 134.5336609
2: -58.6968231, 69.1731873, -58.5425873, 69.1635132, -127.8603210, 127.7157745
3: -67.7949066, 78.6890335, -67.7027740, 78.6829681, -146.4778748, 146.3918152
4: -75.3903809, 81.5034332, -75.2439270, 81.4966049, -156.8869781, 156.7473602
5: -65.5681763, 76.1042023, -65.4837723, 76.0847168, -141.6528931, 141.5879822
6: -109.7375336, 78.6302032, -109.7224884, 78.5672913, -188.3048248, 188.3526611
7: -78.9402313, 72.7338943, -78.8158340, 72.7383423, -151.6785736, 151.5497131
8: -87.3666611, 100.5165329, -87.1404114, 100.5219727, -187.8886261, 187.6569519
9: -75.8398743, 75.5683365, -75.7170334, 75.5497665, -151.3896484, 151.2853699
10: -107.5026932, 101.2503738, -107.3720093, 101.2323608, -208.7350464, 208.6223755
11: -103.3180618, 64.3176117, -103.2911606, 64.2509003, -167.5689392, 167.6087494
12: -101.9382248, 80.5092621, -101.9439850, 80.4372177, -182.3754120, 182.4532471
13: -110.9347992, 107.9001923, -110.8388367, 107.8548431, -218.7896423, 218.7390289
14: -160.6732178, 91.0830994, -160.5475311, 91.0815506, -251.7547607, 251.6306305
15: -88.5716629, 76.4760284, -88.4897461, 76.4411469, -165.0128021, 164.9657745
16: -109.9810104, 78.5088043, -109.8943024, 78.4928131, -188.4738159, 188.4031067
17: -156.3724518, 83.9925613, -156.3496857, 83.9786682, -240.3511047, 240.3422241
18: -104.8824768, 77.3869171, -104.8701324, 77.2982788, -182.1807556, 182.2570496
19: -77.9978333, 51.5865860, -77.9765625, 51.4911728, -129.4890137, 129.5631409
20: -75.5236359, 56.7407036, -75.5018921, 56.6365051, -132.1601410, 132.2425995
21: -98.1934509, 59.6746445, -98.1609344, 59.5198936, -157.7133331, 157.8355713
22: -100.1332245, 61.9574280, -100.1268311, 61.7987366, -161.9319611, 162.0842590
23: -78.3269348, 66.0670776, -78.2987366, 65.9834290, -144.3103638, 144.3658142
24: -96.7093353, 67.1888962, -96.6857758, 67.0781479, -163.7874756, 163.8746643
25: -83.8053207, 68.4017487, -83.7809982, 68.2102814, -152.0155945, 152.1827393
26: -114.8003387, 88.1626968, -114.7647095, 88.0457535, -202.8460846, 202.9273987
27: -98.4514923, 68.6610031, -98.4285812, 68.5621185, -167.0136108, 167.0895844
28: -77.6990433, 67.4731827, -77.6702881, 67.3482056, -145.0472412, 145.1434631
29: -103.8724823, 58.5234947, -103.8803024, 58.4218216, -162.2943115, 162.4037781
30: -98.1581650, 75.4167709, -98.1259766, 75.2284622, -173.3865967, 173.5427246
31: -102.9884338, 70.9766312, -102.9549637, 70.8273087, -173.8157349, 173.9315948
32: -108.8692551, 69.1462021, -108.8478851, 69.1043854, -177.9736176, 177.9940796
33: -135.0639038, 93.3521271, -135.0051575, 93.2268982, -228.2908020, 228.3572845
34: -111.4769440, 68.9112015, -111.4589539, 68.7824707, -180.2593994, 180.3701477
35: -111.3249969, 74.5898438, -111.3096161, 74.4516220, -185.7766113, 185.8994446
36: -114.7718430, 73.9154053, -114.7746124, 73.7881470, -188.5599976, 188.6900177
37: -159.6834412, 73.2539520, -159.6463318, 73.2289886, -232.9124298, 232.9002838
38: -135.1256256, 86.6000290, -135.1094360, 86.5035324, -221.6291504, 221.7094727
39: -151.2258911, 90.2828217, -151.1705627, 90.2738190, -241.4996643, 241.4533539
40: -124.3878937, 69.2069244, -124.3413696, 69.1849823, -193.5728760, 193.5482788
41: -108.4489441, 79.3342896, -108.4309921, 79.3124084, -187.7613525, 187.7652740
42: -79.2974854, 68.9847794, -79.2868347, 68.9648819, -148.2623444, 148.2715912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 721

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6313845
time: 145.81 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6313845
time: 121.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -127.0321198, 88.7909393, -127.2812347, 89.1101685, -216.1422882, 216.0721741
1: -66.4628906, 67.8091278, -66.6211853, 68.0358124, -134.4987030, 134.4303131
2: -58.5061111, 68.9695663, -58.6640854, 69.1975174, -127.7036285, 127.6336517
3: -67.6635132, 78.5169525, -67.8357391, 78.7478485, -146.4113617, 146.3526917
4: -75.1904144, 81.3261261, -75.4268951, 81.5869217, -156.7773132, 156.7530212
5: -65.4455185, 75.8912354, -65.5786285, 76.1319427, -141.5774536, 141.4698639
6: -109.5817795, 78.4955139, -109.7783356, 78.6720886, -188.2538605, 188.2738495
7: -78.7602692, 72.5573425, -78.9360046, 72.8063431, -151.5666199, 151.4933472
8: -87.0827026, 100.2550354, -87.3970032, 100.6664124, -187.7491150, 187.6520386
9: -75.6638489, 75.3511887, -75.8243408, 75.5854950, -151.2493286, 151.1755371
10: -107.2894135, 101.0580444, -107.4554672, 101.2711258, -208.5605469, 208.5135193
11: -103.0766907, 64.1966782, -103.3899612, 64.3976059, -167.4742889, 167.5866394
12: -101.8447113, 80.3280182, -102.0199661, 80.5736694, -182.4183807, 182.3479919
13: -110.7567368, 107.7074127, -110.9970398, 107.9724350, -218.7291718, 218.7044373
14: -160.4395294, 90.8248825, -160.6632385, 91.1076889, -251.5472107, 251.4881287
15: -88.4237518, 76.2907562, -88.6210251, 76.4719849, -164.8957367, 164.9117737
16: -109.7679520, 78.3265228, -109.9851151, 78.5756378, -188.3435974, 188.3116455
17: -156.1994171, 83.7540131, -156.5038147, 84.0645905, -240.2639771, 240.2578278
18: -104.7429962, 77.2456360, -105.0416183, 77.5415802, -182.2845459, 182.2872467
19: -77.7667694, 51.4596825, -78.0285568, 51.5650635, -129.3318329, 129.4882202
20: -75.2689209, 56.5845795, -75.5116730, 56.6900902, -131.9589996, 132.0962524
21: -97.8917313, 59.4772758, -98.2087250, 59.6132736, -157.5050049, 157.6860046
22: -99.8267365, 61.7533188, -100.1725998, 61.8725281, -161.6992645, 161.9259186
23: -78.1289978, 65.9348450, -78.3799744, 66.1306915, -144.2596893, 144.3148193
24: -96.4231110, 67.0421143, -96.7744751, 67.2300110, -163.6531067, 163.8165894
25: -83.5043182, 68.1592255, -83.8180389, 68.2985916, -151.8029175, 151.9772644
26: -114.6172562, 87.9899521, -114.8365555, 88.1784897, -202.7957458, 202.8265076
27: -98.2161407, 68.5211334, -98.4900818, 68.6576538, -166.8737946, 167.0112152
28: -77.4684601, 67.3118286, -77.7096252, 67.4531326, -144.9216003, 145.0214539
29: -103.6076355, 58.3774033, -103.9633713, 58.5362015, -162.1438141, 162.3407745
30: -97.8342133, 75.1719055, -98.1894836, 75.3876801, -173.2218933, 173.3613892
31: -102.6776123, 70.7795715, -103.0544586, 70.9920654, -173.6696777, 173.8340302
32: -108.7191696, 69.0255966, -108.8874664, 69.1878204, -177.9069824, 177.9130554
33: -134.7187347, 93.1598053, -135.0204773, 93.2838669, -228.0025940, 228.1802826
34: -111.2952576, 68.7297134, -111.5608673, 68.9337158, -180.2289581, 180.2905731
35: -111.1008453, 74.3965454, -111.3891983, 74.5769043, -185.6777344, 185.7857361
36: -114.5930405, 73.7369003, -114.8092270, 73.8596649, -188.4526825, 188.5461273
37: -159.4245911, 73.1774445, -159.7541504, 73.3667603, -232.7913513, 232.9315796
38: -134.9167175, 86.4147339, -135.1407471, 86.5696716, -221.4863892, 221.5554810
39: -150.9116821, 90.2012024, -151.2134247, 90.3360748, -241.2477417, 241.4146271
40: -124.1972122, 69.0485001, -124.4009247, 69.2332611, -193.4304810, 193.4494324
41: -108.3512115, 79.2657776, -108.4988861, 79.4243774, -187.7755890, 187.7646484
42: -79.2149811, 68.8815460, -79.3192749, 69.0177536, -148.2327271, 148.2008209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=750, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 721

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.5164843
time: 166.30 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.5164843
time: 120.57 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -127.3024445, 89.1476212, -127.3003082, 89.1924133, -216.4948578, 216.4479370
1: -66.6384354, 68.0399475, -66.6308899, 68.0885620, -134.7269897, 134.6708374
2: -58.6999588, 69.1885681, -58.6717873, 69.2479095, -127.9478607, 127.8603363
3: -67.7975769, 78.7100143, -67.8441467, 78.7915649, -146.5891418, 146.5541687
4: -75.3939514, 81.5328217, -75.4386292, 81.6335144, -157.0274658, 156.9714355
5: -65.5716934, 76.1234894, -65.5868683, 76.1834335, -141.7551270, 141.7103577
6: -109.7515564, 78.6345596, -109.8148193, 78.6847839, -188.4363251, 188.4493713
7: -78.9445801, 72.7567596, -78.9475250, 72.8523102, -151.7968903, 151.7042847
8: -87.3696136, 100.5627670, -87.4095993, 100.7380295, -188.1076355, 187.9723663
9: -75.8433990, 75.5844269, -75.8364410, 75.6387939, -151.4821777, 151.4208679
10: -107.5080185, 101.2571182, -107.4738083, 101.3121948, -208.8202209, 208.7309265
11: -103.3490372, 64.3209152, -103.4477081, 64.4097748, -167.7587891, 167.7686005
12: -101.9564362, 80.5163574, -102.0423965, 80.5984116, -182.5548401, 182.5587463
13: -110.9394608, 107.9286575, -111.0151825, 108.0180435, -218.9575043, 218.9438477
14: -160.6825562, 91.0881195, -160.6841431, 91.1669159, -251.8494568, 251.7722473
15: -88.5768051, 76.4884949, -88.6352158, 76.5136337, -165.0904236, 165.1236877
16: -109.9983368, 78.5115509, -110.0118790, 78.6140060, -188.6123352, 188.5234222
17: -156.3989410, 83.9968643, -156.5300903, 84.1182251, -240.5171356, 240.5269470
18: -104.9217606, 77.3898163, -105.0803833, 77.5530472, -182.4748077, 182.4701996
19: -78.0167999, 51.5884895, -78.0835266, 51.5715408, -129.5883484, 129.6719971
20: -75.5293350, 56.7448769, -75.5682678, 56.6988487, -132.2281799, 132.3131409
21: -98.2135849, 59.6777878, -98.2794189, 59.6223602, -157.8359375, 157.9571991
22: -100.1499786, 61.9589615, -100.2447510, 61.8820953, -162.0320740, 162.2037048
23: -78.3525848, 66.0706482, -78.4275665, 66.1417999, -144.4943848, 144.4982147
24: -96.7366867, 67.1906815, -96.8454132, 67.2375946, -163.9742737, 164.0361023
25: -83.8228607, 68.4038162, -83.8897095, 68.3093872, -152.1322479, 152.2935181
26: -114.8165970, 88.1676636, -114.8753433, 88.1905060, -203.0070801, 203.0429993
27: -98.4674759, 68.6638336, -98.5452728, 68.6668625, -167.1343384, 167.2091064
28: -77.7156448, 67.4758606, -77.7636414, 67.4609222, -145.1765747, 145.2394867
29: -103.8942108, 58.5250053, -104.0272827, 58.5458183, -162.4400177, 162.5522766
30: -98.1803207, 75.4212189, -98.2665482, 75.4006348, -173.5809326, 173.6877747
31: -103.0219498, 70.9788361, -103.1313095, 71.0019608, -174.0238953, 174.1101379
32: -108.8770065, 69.1531296, -108.9191742, 69.2053833, -178.0823975, 178.0722961
33: -135.0741425, 93.3563385, -135.0932770, 93.2977829, -228.3719177, 228.4496155
34: -111.5021667, 68.9152527, -111.6037827, 68.9448471, -180.4470215, 180.5190430
35: -111.3478317, 74.5938416, -111.4410706, 74.5885468, -185.9363403, 186.0349121
36: -114.7849426, 73.9176102, -114.8521805, 73.8709641, -188.6559143, 188.7697906
37: -159.7096252, 73.2566147, -159.8095398, 73.3749695, -233.0845947, 233.0661621
38: -135.1354370, 86.6060333, -135.1887817, 86.5889740, -221.7244110, 221.7948151
39: -151.2343292, 90.2972107, -151.2752075, 90.3501205, -241.5844421, 241.5724182
40: -124.3971405, 69.2114410, -124.4362259, 69.2646942, -193.6618347, 193.6476440
41: -108.4634247, 79.3386612, -108.5187225, 79.4249954, -187.8883972, 187.8573914
42: -79.3022232, 68.9899445, -79.3361435, 69.0304108, -148.3326416, 148.3260803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 721

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6313845
time: 150.32 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7273055, upper bound: 121.6313845
time: 153.15 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -127.2425385, 88.8981171, -127.1217575, 89.0522690, -216.2947693, 216.0198669
1: -66.5940475, 67.8692322, -66.5033646, 67.9673386, -134.5613861, 134.3725891
2: -58.6512375, 69.0435486, -58.5375671, 69.1247253, -127.7759399, 127.5811157
3: -67.8400269, 78.6201019, -67.6965332, 78.6555634, -146.4955750, 146.3166199
4: -75.4242859, 81.4466858, -75.2351227, 81.4723129, -156.8966064, 156.6818085
5: -65.5478058, 75.9741898, -65.4783401, 76.0460663, -141.5938721, 141.4525299
6: -109.6789856, 78.6171875, -109.6959839, 78.5581512, -188.2371216, 188.3131714
7: -78.9266281, 72.6723480, -78.8079834, 72.7131195, -151.6397400, 151.4803314
8: -87.4449921, 100.4516296, -87.1301651, 100.4886856, -187.9336853, 187.5817871
9: -75.7853088, 75.4281311, -75.7074585, 75.5063171, -151.2916260, 151.1355896
10: -107.3957214, 101.1658859, -107.3579788, 101.1970825, -208.5928040, 208.5238647
11: -103.2116776, 64.3409729, -103.2537460, 64.2411194, -167.4527893, 167.5947113
12: -101.9580383, 80.5483856, -101.9395981, 80.4185104, -182.3765564, 182.4879761
13: -110.9180756, 107.8367004, -110.8246841, 107.8240814, -218.7421570, 218.6613770
14: -160.6114807, 90.9608841, -160.5317535, 91.0266571, -251.6381378, 251.4926453
15: -88.5747681, 76.3456879, -88.4801025, 76.4056702, -164.9804077, 164.8257904
16: -109.9201736, 78.5086594, -109.8853760, 78.4565430, -188.3767090, 188.3940430
17: -156.4491577, 83.9947739, -156.3561707, 83.9292145, -240.3783722, 240.3509521
18: -104.9289627, 77.5490341, -104.8616638, 77.2892303, -182.2182007, 182.4106750
19: -77.8684235, 51.5348320, -77.9355469, 51.4865990, -129.3550262, 129.4703674
20: -75.3827057, 56.6498909, -75.4490967, 56.6277161, -132.0103760, 132.0989838
21: -97.9961624, 59.5652542, -98.1030731, 59.5135231, -157.5096893, 157.6683197
22: -99.9703522, 61.8516960, -100.0707092, 61.7901840, -161.7605286, 161.9224091
23: -78.2368011, 66.0991974, -78.2683258, 65.9750366, -144.2118378, 144.3675232
24: -96.5721664, 67.2223969, -96.6342621, 67.0720520, -163.6442108, 163.8566589
25: -83.6067429, 68.2611084, -83.7216263, 68.2007675, -151.8075104, 151.9827271
26: -114.7232895, 88.1363831, -114.7357407, 88.0382385, -202.7615356, 202.8721313
27: -98.3445129, 68.6159897, -98.3840332, 68.5551071, -166.8996277, 167.0000000
28: -77.5515289, 67.4165115, -77.6262054, 67.3423386, -144.8938599, 145.0427246
29: -103.7984848, 58.5369110, -103.8399658, 58.4129791, -162.2114563, 162.3768616
30: -97.9580536, 75.3555527, -98.0629044, 75.2187347, -173.1767883, 173.4184570
31: -102.8290024, 70.9686432, -102.9006042, 70.8193970, -173.6483917, 173.8692474
32: -108.8009186, 69.1332092, -108.8219757, 69.0924988, -177.8934021, 177.9551697
33: -134.8062286, 93.2341537, -134.9382477, 93.2163239, -228.0225525, 228.1723938
34: -111.4388199, 68.9179840, -111.4372559, 68.7751389, -180.2139587, 180.3552399
35: -111.2430954, 74.5688705, -111.2785721, 74.4434357, -185.6865234, 185.8474426
36: -114.6876831, 73.8367767, -114.7444992, 73.7787552, -188.4664307, 188.5812683
37: -159.5990143, 73.3559036, -159.6139526, 73.2229004, -232.8219147, 232.9698486
38: -135.0086823, 86.5108719, -135.0683899, 86.4886246, -221.4973145, 221.5792542
39: -151.0482330, 90.3024597, -151.1155396, 90.2752686, -241.3235016, 241.4179840
40: -124.3188019, 69.1322327, -124.3135452, 69.1568832, -193.4756622, 193.4457703
41: -108.4500351, 79.3855438, -108.4228516, 79.3150864, -187.7651062, 187.8083801
42: -79.2796478, 68.9645309, -79.2733765, 68.9565430, -148.2361908, 148.2379150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=750, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 721

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6126703
time: 171.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6126703
time: 152.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 326.15 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.5164843
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.5164843
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6313845
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6313845
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.5164843
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.5164843
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6313845
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.7273055, upper bound: 121.6313845
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6126703
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 326.15
Output dim: 8, lower bound: -121.5157950, upper bound: 121.6126703
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 326.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.7296275
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 326.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6148778
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 326.15
Output dim: 8, lower bound: -121.6456597, upper bound: 121.6336356
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=187.93016052246094
rel_dist={8: [-121.77351111190006, 121.77351109658267]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1785

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1717766, upper bound: 120.0879213
time: 119.73 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1717766, upper bound: 120.1717764
time: 130.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 250.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 250.34
Output dim: 8, lower bound: -120.1717766, upper bound: 120.0879213
IS_A2, status: Status.UNKNOWN, split count: 1, time: 250.34
Output dim: 8, lower bound: -120.1717766, upper bound: 120.1717764

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -127.1311264, 89.1436310, -127.1894760, 89.1579666, -216.2890930, 216.3330994
1: -66.5138092, 68.0384521, -66.5319061, 68.0759125, -134.5897217, 134.5703430
2: -58.5468636, 69.1890488, -58.5582161, 69.2360229, -127.7828827, 127.7472534
3: -67.7071228, 78.7096252, -67.7167358, 78.7704163, -146.4775391, 146.4263611
4: -75.2487946, 81.5308990, -75.2612305, 81.6165924, -156.8653870, 156.7921143
5: -65.4886627, 76.1151276, -65.5009613, 76.1654053, -141.6540680, 141.6160889
6: -109.7483292, 78.5794678, -109.7891922, 78.5935593, -188.3418579, 188.3686523
7: -78.8203201, 72.7568512, -78.8379593, 72.8326569, -151.6529846, 151.5948181
8: -87.1449585, 100.5661545, -87.1555939, 100.7093887, -187.8543243, 187.7217407
9: -75.7243347, 75.5817566, -75.7346344, 75.6200562, -151.3443909, 151.3163910
10: -107.3807678, 101.2442780, -107.3990936, 101.2663422, -208.6470642, 208.6433411
11: -103.3352585, 64.2569122, -103.4250717, 64.2673035, -167.6025543, 167.6819763
12: -101.9506149, 80.4471130, -102.0183487, 80.4709625, -182.4215698, 182.4654541
13: -110.8471680, 107.9141769, -110.8632355, 107.9751587, -218.8223267, 218.7774048
14: -160.5510559, 91.0945892, -160.5917664, 91.1108246, -251.6618805, 251.6863403
15: -88.4957047, 76.4759598, -88.5143280, 76.5029297, -164.9986267, 164.9902802
16: -109.9002151, 78.5042496, -109.9692688, 78.5123901, -188.4125977, 188.4735107
17: -156.3343353, 83.9851913, -156.4627075, 84.0019989, -240.3363342, 240.4478760
18: -104.9205704, 77.3037720, -105.0341873, 77.3135147, -182.2340851, 182.3379517
19: -78.0098114, 51.4939690, -78.0637360, 51.5015335, -129.5113525, 129.5577087
20: -75.5152817, 56.6305389, -75.5310211, 56.6546898, -132.1699524, 132.1615448
21: -98.2035141, 59.5246239, -98.2564850, 59.5354309, -157.7389526, 157.7811127
22: -100.1405792, 61.8021851, -100.2005920, 61.8077240, -161.9483032, 162.0027771
23: -78.3402786, 65.9895554, -78.4075012, 66.0005646, -144.3408508, 144.3970490
24: -96.7298279, 67.0822754, -96.8025208, 67.0884476, -163.8182678, 163.8847809
25: -83.8112106, 68.2155304, -83.8579712, 68.2223663, -152.0335693, 152.0735016
26: -114.8033524, 88.0512085, -114.8429489, 88.0690231, -202.8723450, 202.8941345
27: -98.4599075, 68.5677719, -98.5022430, 68.5762177, -167.0361328, 167.0700073
28: -77.7048492, 67.3536377, -77.7459488, 67.3618774, -145.0667267, 145.0995789
29: -103.8900681, 58.4268112, -103.9754868, 58.4309883, -162.3210602, 162.4022980
30: -98.1680603, 75.2357712, -98.2248077, 75.2494812, -173.4175415, 173.4605713
31: -103.0106888, 70.8320160, -103.0954132, 70.8400650, -173.8507538, 173.9274292
32: -108.8633575, 69.1159210, -108.8862000, 69.1376648, -178.0010071, 178.0021210
33: -135.0333557, 93.2313156, -135.0565948, 93.2487793, -228.2821350, 228.2879028
34: -111.4851456, 68.7869720, -111.5644836, 68.8035431, -180.2886963, 180.3514557
35: -111.3274078, 74.4565659, -111.4054871, 74.4721985, -185.7995911, 185.8620453
36: -114.7816772, 73.7921448, -114.8292542, 73.8005829, -188.5822601, 188.6213989
37: -159.6750336, 73.2374573, -159.7621307, 73.2475586, -232.9225769, 232.9995728
38: -135.1294861, 86.5081635, -135.1568756, 86.5333710, -221.6628265, 221.6650391
39: -151.1862946, 90.2730103, -151.2131042, 90.3340302, -241.5203247, 241.4861145
40: -124.3573074, 69.1886826, -124.3870087, 69.2087402, -193.5660400, 193.5756836
41: -108.4479294, 79.3339767, -108.4950333, 79.3469849, -187.7949219, 187.8290100
42: -79.2953491, 68.9645996, -79.3097610, 68.9957581, -148.2911072, 148.2743530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=524, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -120.0858501, upper bound: 120.0755598
time: 2693.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -120.0858501, upper bound: 120.0755598
time: 120.97 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -127.3538971, 89.2551880, -127.2028198, 89.1625214, -216.5164185, 216.4580078
1: -66.6497192, 68.1133728, -66.5379181, 68.0885162, -134.7382355, 134.6512909
2: -58.6951294, 69.2781219, -58.5620956, 69.2521896, -127.9473190, 127.8402176
3: -67.8862610, 78.8332672, -67.7198257, 78.7929230, -146.6791840, 146.5531006
4: -75.4861374, 81.6803513, -75.2653351, 81.6478424, -157.1339722, 156.9456787
5: -65.5944061, 76.2170715, -65.5049591, 76.1831512, -141.7775574, 141.7220306
6: -109.8595734, 78.7055054, -109.8032608, 78.5985641, -188.4581299, 188.5087585
7: -78.9908905, 72.8946762, -78.8431854, 72.8612061, -151.8520966, 151.7378540
8: -87.5101776, 100.8090134, -87.1588898, 100.7622986, -188.2724762, 187.9678955
9: -75.8492584, 75.6747284, -75.7381134, 75.6337509, -151.4829865, 151.4128418
10: -107.4923935, 101.3588181, -107.4050903, 101.2742157, -208.7666016, 208.7638855
11: -103.5003204, 64.4044418, -103.4538040, 64.2706528, -167.7709656, 167.8582458
12: -102.0819778, 80.6744537, -102.0433807, 80.4793549, -182.5613098, 182.7178345
13: -111.0131760, 108.0716629, -110.8687668, 107.9959564, -219.0090942, 218.9404144
14: -160.7323914, 91.2356033, -160.5999451, 91.1165543, -251.8489380, 251.8355408
15: -88.6518555, 76.5433502, -88.5206299, 76.5117111, -165.1635742, 165.0639648
16: -110.0696335, 78.6890869, -109.9940720, 78.5152740, -188.5848999, 188.6831665
17: -156.6102295, 84.2303314, -156.5080872, 84.0079346, -240.6181030, 240.7383728
18: -105.1457748, 77.6100159, -105.0759888, 77.3168793, -182.4626465, 182.6860046
19: -78.1299133, 51.5710678, -78.0830231, 51.5042496, -129.6341553, 129.6540833
20: -75.6346283, 56.7000542, -75.5363007, 56.6557503, -132.2903748, 132.2363586
21: -98.3277283, 59.6157417, -98.2743301, 59.5392036, -157.8669281, 157.8900757
22: -100.3006821, 61.9020538, -100.2228622, 61.8091812, -162.1098480, 162.1249084
23: -78.4733353, 66.1574326, -78.4313202, 66.0043030, -144.4776306, 144.5887451
24: -96.9062805, 67.2642975, -96.8297806, 67.0904770, -163.9967346, 164.0940552
25: -83.9311523, 68.3194427, -83.8750458, 68.2242126, -152.1553650, 152.1944885
26: -114.9255447, 88.2026978, -114.8565903, 88.0754318, -203.0009613, 203.0592957
27: -98.6040497, 68.6654434, -98.5171127, 68.5792084, -167.1832581, 167.1825562
28: -77.8043823, 67.4609833, -77.7596741, 67.3646088, -145.1689758, 145.2206573
29: -104.1020508, 58.5877991, -104.0080414, 58.4320869, -162.5341339, 162.5958252
30: -98.3139038, 75.4237823, -98.2441025, 75.2540436, -173.5679321, 173.6678772
31: -103.1951599, 71.0233078, -103.1258621, 70.8427887, -174.0379486, 174.1491699
32: -108.9527740, 69.2304688, -108.8942947, 69.1455536, -178.0983276, 178.1247559
33: -135.1309052, 93.3098450, -135.0647888, 93.2536316, -228.3845215, 228.3746338
34: -111.6538925, 68.9791489, -111.5937500, 68.8088684, -180.4627686, 180.5729065
35: -111.4924774, 74.6328888, -111.4343109, 74.4770660, -185.9695282, 186.0671997
36: -114.8892288, 73.8942490, -114.8467712, 73.8031845, -188.6923981, 188.7410278
37: -159.8751068, 73.4185944, -159.7932434, 73.2505341, -233.1256104, 233.2118378
38: -135.2311096, 86.6103363, -135.1664734, 86.5397644, -221.7708740, 221.7767639
39: -151.3312378, 90.3884811, -151.2225494, 90.3555145, -241.6867523, 241.6110229
40: -124.4881058, 69.2769165, -124.3974457, 69.2137070, -193.7017822, 193.6743469
41: -108.5610733, 79.4581070, -108.5111160, 79.3516006, -187.9126587, 187.9692078
42: -79.3646469, 69.0527725, -79.3144455, 69.0023346, -148.3669739, 148.3672180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=524, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1152
type: B, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.0858501, upper bound: 120.1607279
time: 123.97 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.0858501, upper bound: 120.1607279
time: 198.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 325.24 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 325.24
Output dim: 8, lower bound: -120.0858501, upper bound: 120.0755598
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 325.24
Output dim: 8, lower bound: -120.0858501, upper bound: 120.0755598
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 325.24
Output dim: 8, lower bound: -120.0858501, upper bound: 120.1607279
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 325.24
Output dim: 8, lower bound: -120.0858501, upper bound: 120.1607279

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -127.3343353, 89.2485809, -127.1445312, 89.1425781, -216.4769135, 216.3930969
1: -66.6424866, 68.0919495, -66.5161362, 68.0244141, -134.6669006, 134.6080933
2: -58.6903419, 69.2542267, -58.5477905, 69.1788940, -127.8692322, 127.8020172
3: -67.8822327, 78.8042450, -67.7076111, 78.7036667, -146.5858765, 146.5118561
4: -75.4810486, 81.6392822, -75.2498398, 81.5226517, -157.0036926, 156.8891296
5: -65.5892029, 76.1902161, -65.4891663, 76.1019135, -141.6911163, 141.6793823
6: -109.8375092, 78.6992569, -109.7370911, 78.5795135, -188.4170227, 188.4363403
7: -78.9842072, 72.8623199, -78.8228531, 72.7624207, -151.7466125, 151.6851654
8: -87.5059357, 100.7434998, -87.1460648, 100.5651093, -188.0710449, 187.8895569
9: -75.8442230, 75.6519928, -75.7228546, 75.5640869, -151.4083099, 151.3748474
10: -107.4845123, 101.3489227, -107.3811264, 101.2442551, -208.7287598, 208.7300415
11: -103.4565811, 64.3997498, -103.3171844, 64.2564087, -167.7129822, 167.7169342
12: -102.0561371, 80.6642761, -101.9656296, 80.4485703, -182.5046997, 182.6298981
13: -111.0063019, 108.0318985, -110.8479233, 107.8755112, -218.8818054, 218.8798218
14: -160.7187042, 91.2284088, -160.5587463, 91.0946198, -251.8133240, 251.7871552
15: -88.6445007, 76.5237274, -88.4982529, 76.4519196, -165.0964203, 165.0219727
16: -110.0443268, 78.6851578, -109.9179840, 78.5031815, -188.5475159, 188.6031189
17: -156.5712128, 84.2238312, -156.3887177, 83.9886475, -240.5598450, 240.6125488
18: -105.0901642, 77.6058502, -104.9052124, 77.3042068, -182.3943481, 182.5110626
19: -78.1014175, 51.5681725, -77.9959412, 51.4954453, -129.5968475, 129.5641174
20: -75.6264572, 56.6940689, -75.5114288, 56.6379089, -132.2643585, 132.2055054
21: -98.2967453, 59.6113319, -98.1805878, 59.5257301, -157.8224792, 157.7919159
22: -100.2762985, 61.8998680, -100.1494141, 61.8026161, -162.0789185, 162.0492859
23: -78.4374237, 66.1524048, -78.3206635, 65.9889297, -144.4263458, 144.4730530
24: -96.8672714, 67.2616730, -96.7111435, 67.0825577, -163.9498138, 163.9728088
25: -83.9062347, 68.3164749, -83.7989349, 68.2151871, -152.1214294, 152.1153870
26: -114.9010315, 88.1953735, -114.7820740, 88.0537415, -202.9547729, 202.9774475
27: -98.5803223, 68.6614380, -98.4449387, 68.5669785, -167.1472778, 167.1063843
28: -77.7800293, 67.4571075, -77.6851501, 67.3529892, -145.1330109, 145.1422577
29: -104.0700989, 58.5856514, -103.9101715, 58.4255714, -162.4956512, 162.4958191
30: -98.2818527, 75.4175415, -98.1464767, 75.2349548, -173.5168152, 173.5640259
31: -103.1478958, 71.0200882, -102.9838562, 70.8330383, -173.9809265, 174.0039368
32: -108.9408875, 69.2206345, -108.8582306, 69.1154709, -178.0563354, 178.0788574
33: -135.1156921, 93.3034286, -135.0184937, 93.2340851, -228.3497772, 228.3219147
34: -111.6181641, 68.9728165, -111.4842834, 68.7894440, -180.4075928, 180.4570923
35: -111.4600754, 74.6267776, -111.3350449, 74.4584656, -185.9185181, 185.9618225
36: -114.8709488, 73.8908234, -114.7918320, 73.7928619, -188.6638184, 188.6826477
37: -159.8369293, 73.4144287, -159.6758728, 73.2379761, -233.0749054, 233.0903015
38: -135.2165833, 86.6010742, -135.1222839, 86.5116806, -221.7282715, 221.7233582
39: -151.3190002, 90.3675232, -151.1854553, 90.2924118, -241.6113739, 241.5529480
40: -124.4738770, 69.2698975, -124.3538971, 69.1925201, -193.6663971, 193.6237946
41: -108.5395432, 79.4517822, -108.4462662, 79.3323822, -187.8719177, 187.8980408
42: -79.3578033, 69.0442505, -79.2936249, 68.9763947, -148.3341980, 148.3378601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 610

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -120.0828809, upper bound: 120.0529757
time: 152.76 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.0828809, upper bound: 120.1577163
time: 130.57 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -127.3503265, 89.2542801, -127.3129196, 89.2036743, -216.5540009, 216.5671997
1: -66.6485977, 68.1106186, -66.6382523, 68.1011200, -134.7497253, 134.7488708
2: -58.6943855, 69.2734375, -58.6770515, 69.2621002, -127.9564819, 127.9504776
3: -67.8855591, 78.8298798, -67.8489227, 78.8119965, -146.6975555, 146.6788025
4: -75.4854889, 81.6758423, -75.4445114, 81.6594315, -157.1449280, 157.1203613
5: -65.5935974, 76.2146225, -65.5922394, 76.2005539, -141.7941589, 141.8068542
6: -109.8555756, 78.7048187, -109.8293762, 78.6970825, -188.5526581, 188.5341797
7: -78.9896317, 72.8917236, -78.9545135, 72.8763428, -151.8659668, 151.8462372
8: -87.5096741, 100.8030777, -87.4152222, 100.7811508, -188.2908325, 188.2182922
9: -75.8486786, 75.6725922, -75.8421783, 75.6530685, -151.5017395, 151.5147705
10: -107.4912415, 101.3574600, -107.4829102, 101.3239136, -208.8151245, 208.8403625
11: -103.4944763, 64.4038696, -103.4736786, 64.4152145, -167.9096985, 167.8775330
12: -102.0793457, 80.6732788, -102.0640411, 80.6097107, -182.6890564, 182.7373199
13: -111.0122604, 108.0679474, -111.0242233, 108.0386887, -219.0509491, 219.0921631
14: -160.7308502, 91.2347412, -160.6953430, 91.1798325, -251.9106598, 251.9300842
15: -88.6510315, 76.5399170, -88.6436844, 76.5243988, -165.1754303, 165.1835938
16: -110.0659332, 78.6885986, -110.0351028, 78.6243439, -188.6902771, 188.7236938
17: -156.6043701, 84.2295303, -156.5689392, 84.1283035, -240.7326660, 240.7984619
18: -105.1407471, 77.6094513, -105.1154327, 77.5589142, -182.6996613, 182.7248840
19: -78.1245117, 51.5706558, -78.1012726, 51.5758362, -129.7003479, 129.6719055
20: -75.6337128, 56.6994743, -75.5777359, 56.7003212, -132.3340302, 132.2772064
21: -98.3218079, 59.6153488, -98.2980652, 59.6281853, -157.9499817, 157.9134216
22: -100.2973785, 61.9017868, -100.2670288, 61.8859482, -162.1833191, 162.1687927
23: -78.4697571, 66.1568832, -78.4495010, 66.1472549, -144.6170044, 144.6063843
24: -96.9027557, 67.2638702, -96.8710785, 67.2419052, -164.1446533, 164.1349487
25: -83.9288025, 68.3190536, -83.9076385, 68.3142090, -152.2429962, 152.2266846
26: -114.9218826, 88.2018661, -114.8926392, 88.1986160, -203.1204834, 203.0944824
27: -98.6004486, 68.6650238, -98.5612106, 68.6717224, -167.2721710, 167.2262268
28: -77.8011780, 67.4605408, -77.7783051, 67.4657135, -145.2668762, 145.2388458
29: -104.0966187, 58.5875473, -104.0568695, 58.5495224, -162.6461487, 162.6444092
30: -98.3102417, 75.4231339, -98.2866211, 75.4070587, -173.7172699, 173.7097473
31: -103.1898956, 71.0228882, -103.1594391, 71.0076523, -174.1975403, 174.1823273
32: -108.9507980, 69.2295456, -108.9293747, 69.2164764, -178.1672668, 178.1589050
33: -135.1287842, 93.3086929, -135.1066284, 93.3048935, -228.4336853, 228.4153137
34: -111.6506195, 68.9778671, -111.6290207, 68.9517822, -180.6024017, 180.6068726
35: -111.4895020, 74.6318512, -111.4664993, 74.5953827, -186.0848846, 186.0983582
36: -114.8873291, 73.8935776, -114.8690643, 73.8755798, -188.7628937, 188.7626343
37: -159.8695068, 73.4177704, -159.8373413, 73.3838272, -233.2533264, 233.2551117
38: -135.2290649, 86.6087799, -135.2015533, 86.5967407, -221.8258057, 221.8103027
39: -151.3298340, 90.3854370, -151.2900696, 90.3685532, -241.6983948, 241.6755066
40: -124.4857788, 69.2756271, -124.4487305, 69.2720490, -193.7578278, 193.7243347
41: -108.5577393, 79.4573669, -108.5333633, 79.4449463, -188.0026855, 187.9907227
42: -79.3637238, 69.0509033, -79.3427658, 69.0419312, -148.4056396, 148.3936768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=523, inp2_unstable=523, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=751, inp2_unstable=751, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1152
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 610

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -120.0828809, upper bound: 120.0529757
time: 126.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.0828809, upper bound: 120.1577163
time: 115.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 244.84 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 244.84
Output dim: 8, lower bound: -120.0828809, upper bound: 120.0529757
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 244.84
Output dim: 8, lower bound: -120.0828809, upper bound: 120.1577163
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 244.84
Output dim: 8, lower bound: -120.0828809, upper bound: 120.0529757
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 244.84
Output dim: 8, lower bound: -120.0828809, upper bound: 120.1577163
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=187.93016052246094
rel_dist={8: [-120.19734924920294, 120.19734925085426]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 12450.81 seconds
