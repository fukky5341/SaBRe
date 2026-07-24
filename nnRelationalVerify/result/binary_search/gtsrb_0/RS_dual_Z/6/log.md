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
execution time: IAR + LP analysis = 2.89 + 180.13 = 183.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -127.4375409, upper bound: 127.4375409


# Binary Search by BASE starts (time budget: 17816.98 seconds, max iter: 100)

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
Binary search time: 1834.35 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 15982.64 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2930521, upper bound: 124.4440983
time: 105.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.4440983, upper bound: 124.2930521
time: 121.17 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 227.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 227.13
Output dim: 8, lower bound: -124.2930521, upper bound: 124.4440983
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 227.13
Output dim: 8, lower bound: -124.4440983, upper bound: 124.2930521

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

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.1157502, upper bound: 124.2682278
time: 127.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.1187661, upper bound: 124.2653032
time: 102.94 seconds

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
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2653032, upper bound: 124.1187661
time: 118.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2682278, upper bound: 124.1157502
time: 135.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 256.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 256.86
Output dim: 8, lower bound: -124.1157502, upper bound: 124.2682278
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 256.86
Output dim: 8, lower bound: -124.1187661, upper bound: 124.2653032
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 256.86
Output dim: 8, lower bound: -124.2653032, upper bound: 124.1187661
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 256.86
Output dim: 8, lower bound: -124.2682278, upper bound: 124.1157502

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

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -123.9667820, upper bound: 124.2640248
time: 118.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.1114886, upper bound: 124.1213171
time: 1580.01 seconds

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

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -123.9698665, upper bound: 124.2610996
time: 134.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.1145065, upper bound: 124.1183390
time: 213.48 seconds

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

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.1183390, upper bound: 124.1145065
time: 185.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2610996, upper bound: 123.9698665
time: 142.14 seconds

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

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.1213171, upper bound: 124.1114886
time: 127.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.2640248, upper bound: 123.9667820
time: 135.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 266.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 266.13
Output dim: 8, lower bound: -123.9667820, upper bound: 124.2640248
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 266.13
Output dim: 8, lower bound: -124.1114886, upper bound: 124.1213171
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 266.13
Output dim: 8, lower bound: -123.9698665, upper bound: 124.2610996
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 266.13
Output dim: 8, lower bound: -124.1145065, upper bound: 124.1183390
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 266.13
Output dim: 8, lower bound: -124.1183390, upper bound: 124.1145065
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 266.13
Output dim: 8, lower bound: -124.2610996, upper bound: 123.9698665
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 266.13
Output dim: 8, lower bound: -124.1213171, upper bound: 124.1114886
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 266.13
Output dim: 8, lower bound: -124.2640248, upper bound: 123.9667820

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

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -123.9159077, upper bound: 124.1911432
time: 107.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -123.8940674, upper bound: 124.2126126
time: 140.70 seconds

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

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.0605167, upper bound: 124.0487020
time: 136.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -124.0388102, upper bound: 124.0703133
time: 688.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 827.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.24
Output dim: 8, lower bound: -123.9159077, upper bound: 124.1911432
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.24
Output dim: 8, lower bound: -123.8940674, upper bound: 124.2126126
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.24
Output dim: 8, lower bound: -124.0605167, upper bound: 124.0487020
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.24
Output dim: 8, lower bound: -124.0388102, upper bound: 124.0703133
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 827.24
Output dim: 8, lower bound: -123.9698665, upper bound: 124.2610996
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 827.24
Output dim: 8, lower bound: -124.1145065, upper bound: 124.1183390
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 827.24
Output dim: 8, lower bound: -124.1183390, upper bound: 124.1145065
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 827.24
Output dim: 8, lower bound: -124.2610996, upper bound: 123.9698665
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 827.24
Output dim: 8, lower bound: -124.1213171, upper bound: 124.1114886
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 827.24
Output dim: 8, lower bound: -124.2640248, upper bound: 123.9667820
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=187.93016052246094
rel_dist={8: [-124.44835295992456, 124.44835294926614]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6438998, upper bound: 121.7700274
time: 136.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.7700275, upper bound: 121.6438998
time: 134.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 270.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 270.72
Output dim: 8, lower bound: -121.6438998, upper bound: 121.7700274
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 270.72
Output dim: 8, lower bound: -121.7700275, upper bound: 121.6438998

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
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5163366, upper bound: 121.6481169
time: 150.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5227095, upper bound: 121.6418918
time: 139.94 seconds

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
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6418918, upper bound: 121.5227095
time: 140.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6481170, upper bound: 121.5163365
time: 156.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 299.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 299.82
Output dim: 8, lower bound: -121.5163366, upper bound: 121.6481169
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 299.82
Output dim: 8, lower bound: -121.5227095, upper bound: 121.6418918
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 299.82
Output dim: 8, lower bound: -121.6418918, upper bound: 121.5227095
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 299.82
Output dim: 8, lower bound: -121.6481170, upper bound: 121.5163365

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

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.3963626, upper bound: 121.6446428
time: 204.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5127953, upper bound: 121.5293797
time: 135.42 seconds

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

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.4027887, upper bound: 121.6384174
time: 149.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5191692, upper bound: 121.5230341
time: 131.07 seconds

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
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5230341, upper bound: 121.5191692
time: 150.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6384174, upper bound: 121.4027886
time: 142.36 seconds

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

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.5293797, upper bound: 121.5127953
time: 172.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.6446428, upper bound: 121.3963626
time: 112.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 287.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 287.08
Output dim: 8, lower bound: -121.3963626, upper bound: 121.6446428
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 287.08
Output dim: 8, lower bound: -121.5127953, upper bound: 121.5293797
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 287.08
Output dim: 8, lower bound: -121.4027887, upper bound: 121.6384174
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 287.08
Output dim: 8, lower bound: -121.5191692, upper bound: 121.5230341
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 287.08
Output dim: 8, lower bound: -121.5230341, upper bound: 121.5191692
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 287.08
Output dim: 8, lower bound: -121.6384174, upper bound: 121.4027886
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 287.08
Output dim: 8, lower bound: -121.5293797, upper bound: 121.5127953
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 287.08
Output dim: 8, lower bound: -121.6446428, upper bound: 121.3963626

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.3630765, upper bound: 121.5843730
time: 187.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.3354891, upper bound: 121.6116175
time: 143.25 seconds

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

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.4796452, upper bound: 121.4688099
time: 113.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.4521978, upper bound: 121.4961797
time: 162.30 seconds

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.3695709, upper bound: 121.5780243
time: 130.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.3420463, upper bound: 121.6053145
time: 114.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.4860286, upper bound: 121.4623725
time: 1833.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -121.4586570, upper bound: 121.4898207
time: 158.65 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1994.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1994.80
Output dim: 8, lower bound: -121.3630765, upper bound: 121.5843730
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1994.80
Output dim: 8, lower bound: -121.3354891, upper bound: 121.6116175
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1994.80
Output dim: 8, lower bound: -121.4796452, upper bound: 121.4688099
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1994.80
Output dim: 8, lower bound: -121.4521978, upper bound: 121.4961797
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1994.80
Output dim: 8, lower bound: -121.3695709, upper bound: 121.5780243
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1994.80
Output dim: 8, lower bound: -121.3420463, upper bound: 121.6053145
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1994.80
Output dim: 8, lower bound: -121.4860286, upper bound: 121.4623725
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1994.80
Output dim: 8, lower bound: -121.4586570, upper bound: 121.4898207
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1994.80
Output dim: 8, lower bound: -121.5230341, upper bound: 121.5191692
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1994.80
Output dim: 8, lower bound: -121.6384174, upper bound: 121.4027886
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1994.80
Output dim: 8, lower bound: -121.5293797, upper bound: 121.5127953
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1994.80
Output dim: 8, lower bound: -121.6446428, upper bound: 121.3963626
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=187.93016052246094
rel_dist={8: [-121.77351111190006, 121.77351109658267]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.0822709, upper bound: 120.1942502
time: 715.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.1942502, upper bound: 120.0822709
time: 168.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 883.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 883.80
Output dim: 8, lower bound: -120.0822709, upper bound: 120.1942502
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 883.80
Output dim: 8, lower bound: -120.1942502, upper bound: 120.0822709

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
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -119.9689653, upper bound: 120.0889135
time: 124.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -119.9773098, upper bound: 120.0806288
time: 138.46 seconds

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

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -120.0806288, upper bound: 119.9773098
time: 154.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -120.0889135, upper bound: 119.9689653
time: 144.87 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 301.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 301.45
Output dim: 8, lower bound: -119.9689653, upper bound: 120.0889135
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 301.45
Output dim: 8, lower bound: -119.9773098, upper bound: 120.0806288
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 301.45
Output dim: 8, lower bound: -120.0806288, upper bound: 119.9773098
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 301.45
Output dim: 8, lower bound: -120.0889135, upper bound: 119.9689653

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

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -119.8632152, upper bound: 120.0858948
time: 130.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -119.9659059, upper bound: 119.9840357
time: 140.55 seconds

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

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 874

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -119.9840357, upper bound: 119.9659059
time: 157.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -120.0858948, upper bound: 119.8632152
time: 110.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 269.54 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 269.54
Output dim: 8, lower bound: -119.8632152, upper bound: 120.0858948
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 269.54
Output dim: 8, lower bound: -119.9659059, upper bound: 119.9840357
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 269.54
Output dim: 8, lower bound: -119.9840357, upper bound: 119.9659059
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 269.54
Output dim: 8, lower bound: -120.0858948, upper bound: 119.8632152
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=187.93016052246094
rel_dist={8: [-120.19734924920294, 120.19734925085426]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 11844.76 seconds
