## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 52.4281605764
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347)
1: (-68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480)
2: (-61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091)
3: (-70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121)
4: (-76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159)
5: (-68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184)
6: (-108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174)
7: (-80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013)
8: (-89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615)
9: (-75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750)
10: (-111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705)
11: (-105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525)
12: (-110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491)
13: (-108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455)
14: (-167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575)
15: (-88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989)
16: (-111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022)
17: (-159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412)
18: (-107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520)
19: (-80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203)
20: (-76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390)
21: (-100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346)
22: (-103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319)
23: (-82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812)
24: (-101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620)
25: (-87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429)
26: (-116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759)
27: (-101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959)
28: (-80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429)
29: (-107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427)
30: (-101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114)
31: (-107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098)
32: (-105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004)
33: (-139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966)
34: (-118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812)
35: (-116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027)
36: (-113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031)
37: (-167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510)
38: (-134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424)
39: (-157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871)
40: (-124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667)
41: (-112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673)
42: (-79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781)

## BASE Result
execution time: IAR + LP analysis = 2.88 + 151.91 = 154.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -109.1201928, upper bound: 109.1201929


# Binary Search by BASE starts (time budget: 17845.20 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=154.09011840820312
rel_dist={5: [-102.90009993993365, 102.90009993642099]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=154.09011840820312
rel_dist={5: [-98.15387180223486, 98.15387180681547]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=154.09011840820312
rel_dist={5: [-94.06215389057739, 94.06215389943473]}

## Binary Search Result
Binary search time: 577.29 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 17267.91 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.7706313, upper bound: 102.8932220
time: 185.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.7706313, upper bound: 102.8935604
time: 174.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 360.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 360.70
Output dim: 5, lower bound: -102.7706313, upper bound: 102.8932220
IS_A2, status: Status.UNKNOWN, split count: 1, time: 360.70
Output dim: 5, lower bound: -102.7706313, upper bound: 102.8935604

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -131.8210144, 88.8755341, -132.0257721, 88.9280701, -220.7490845, 220.9012756
1: -68.0530472, 68.4902649, -68.2120972, 68.5279999, -136.5810394, 136.7023621
2: -61.2388382, 70.9263306, -61.4792824, 70.9629517, -132.2017822, 132.4056091
3: -69.9746246, 83.4850464, -70.2578735, 83.5433197, -153.5179443, 153.7429199
4: -76.2938843, 82.2709961, -76.5514832, 82.3192520, -158.6131287, 158.8224792
5: -67.9042969, 85.7616119, -68.1675797, 85.8184280, -153.7227173, 153.9291992
6: -108.1409760, 80.6971741, -108.2311859, 80.7864075, -188.9273834, 188.9283447
7: -79.9389725, 80.9567871, -80.1746674, 81.0003967, -160.9393616, 161.1314545
8: -89.4075928, 102.1556931, -89.6483765, 102.2087402, -191.6163177, 191.8040466
9: -75.5651550, 80.3154907, -75.6356049, 80.5044861, -156.0696411, 155.9510956
10: -110.9488144, 104.4469833, -111.0487823, 104.8722916, -215.8211060, 215.4957581
11: -105.1446075, 64.4842682, -105.2250061, 64.7495804, -169.8941956, 169.7092743
12: -110.3833160, 85.4623566, -110.4511871, 85.9255676, -196.3088837, 195.9135132
13: -108.3084030, 109.5601044, -108.3935242, 109.6686935, -217.9770966, 217.9536133
14: -167.4445343, 95.0237350, -167.5706177, 95.3441925, -262.7887268, 262.5943298
15: -88.2537766, 76.2740021, -88.4456329, 76.3547821, -164.6085510, 164.7196350
16: -111.4562149, 80.0732498, -111.5669403, 80.2644806, -191.7206879, 191.6401978
17: -159.4137115, 83.7702408, -159.4931030, 84.0215225, -243.4352417, 243.2633362
18: -107.3642120, 79.2283020, -107.4499130, 79.4109573, -186.7751770, 186.6782227
19: -80.8932495, 48.6582642, -80.9666824, 48.7524414, -129.6456909, 129.6249390
20: -76.8296585, 60.7423935, -76.9061279, 60.8481407, -137.6777954, 137.6485291
21: -100.4394684, 60.2486153, -100.5155029, 60.4150238, -160.8544922, 160.7641144
22: -102.9153290, 62.1113281, -102.9936295, 62.2483292, -165.1636658, 165.1049500
23: -82.0795059, 61.6582718, -82.1452103, 61.7434273, -143.8229065, 143.8034821
24: -101.4433517, 63.1105576, -101.5275879, 63.1554947, -164.5988464, 164.6381531
25: -87.1739120, 66.0765305, -87.2319107, 66.1914902, -153.3653870, 153.3084412
26: -116.7622147, 94.1355286, -116.8487167, 94.4815140, -211.2437286, 210.9842224
27: -101.4418259, 66.8258362, -101.5778732, 66.8638000, -168.3056335, 168.4037170
28: -80.3196564, 67.9175873, -80.3932800, 67.9669952, -148.2866516, 148.3108673
29: -107.2020645, 58.3122139, -107.2696991, 58.4790268, -165.6810913, 165.5819092
30: -101.0540924, 74.2903214, -101.1203537, 74.4388199, -175.4929047, 175.4106750
31: -107.1467133, 67.1315155, -107.2458649, 67.2427826, -174.3894958, 174.3773804
32: -105.5623779, 69.4229431, -105.6350479, 69.5687103, -175.1310883, 175.0579834
33: -139.6206207, 92.2541733, -139.8088074, 92.3288956, -231.9494934, 232.0629883
34: -118.7801971, 64.0901184, -118.9065475, 64.1563644, -182.9365540, 182.9966736
35: -115.8215866, 71.7150040, -115.9789734, 71.7651367, -187.5867004, 187.6939697
36: -113.7192917, 71.6921082, -113.8091507, 71.7456818, -185.4649658, 185.5012512
37: -167.4634857, 74.2872086, -167.5641785, 74.4092102, -241.8726959, 241.8513794
38: -133.9121704, 83.3564224, -134.0541229, 83.4154510, -217.3276062, 217.4105530
39: -157.5838013, 87.6679153, -157.6860046, 87.7293320, -245.3131409, 245.3539124
40: -124.2903137, 72.0222626, -124.3964157, 72.0637360, -196.3540039, 196.4186707
41: -112.1514587, 81.0807648, -112.2380219, 81.1483459, -193.2998047, 193.3187714
42: -79.3160095, 72.4077301, -79.3891449, 72.5612640, -151.8772583, 151.7968750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=604, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.7626060, upper bound: 102.7689310
time: 163.05 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.7626060, upper bound: 102.8826144
time: 184.25 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -132.1593323, 89.2062988, -132.0723724, 88.9396133, -221.0989380, 221.2786560
1: -68.2940826, 68.7705917, -68.2492828, 68.5353394, -136.8294067, 137.0198669
2: -61.5778618, 71.2851944, -61.5381088, 70.9698029, -132.5476532, 132.8232880
3: -70.3527451, 83.9127960, -70.3251724, 83.5545502, -153.9072876, 154.2379761
4: -76.6697540, 82.6348877, -76.6153717, 82.3279419, -158.9976807, 159.2502594
5: -68.2892303, 86.2218170, -68.2365570, 85.8301239, -154.1193390, 154.4583588
6: -108.3680878, 80.8438950, -108.2496338, 80.7905731, -189.1586609, 189.0935211
7: -80.3261566, 81.3207169, -80.2335968, 81.0062561, -161.3324127, 161.5543213
8: -89.7581329, 102.5694275, -89.7077484, 102.2192841, -191.9774170, 192.2771759
9: -75.8355179, 80.6151505, -75.6501694, 80.5498047, -156.3853149, 156.2653198
10: -111.6241302, 105.0287094, -111.0709839, 104.9756622, -216.5997620, 216.0997009
11: -105.6300659, 64.8383026, -105.2411880, 64.8118668, -170.4419250, 170.0794678
12: -111.1356125, 86.0920715, -110.4633026, 86.0410767, -197.1766815, 196.5553741
13: -108.4429474, 109.8420029, -108.3982010, 109.6932755, -218.1362305, 218.2402039
14: -168.0380554, 95.4540558, -167.5953522, 95.4240112, -263.4620361, 263.0493774
15: -88.5243835, 76.5226440, -88.4649200, 76.3718948, -164.8962708, 164.9875641
16: -111.8295441, 80.3562775, -111.5897598, 80.2928543, -192.1224060, 191.9460449
17: -159.9101410, 84.1549072, -159.5074158, 84.0861893, -243.9963226, 243.6623230
18: -107.7147980, 79.4974518, -107.4681091, 79.4527130, -187.1675110, 186.9655609
19: -81.2070465, 48.7980042, -80.9831238, 48.7740288, -129.9810791, 129.7811279
20: -77.1103516, 60.8931122, -76.9236221, 60.8720627, -137.9823914, 137.8167267
21: -100.8555298, 60.4888954, -100.5314178, 60.4553909, -161.3109131, 161.0202942
22: -103.1307068, 62.3346062, -103.0062714, 62.2699623, -165.4006653, 165.3408661
23: -82.3119354, 61.7857437, -82.1592865, 61.7554855, -144.0674133, 143.9450073
24: -101.6200409, 63.1723137, -101.5414658, 63.1565895, -164.7766266, 164.7137756
25: -87.3523483, 66.2720261, -87.2423401, 66.2188492, -153.5711670, 153.5143585
26: -117.3033447, 94.6449585, -116.8638306, 94.5729599, -211.8762970, 211.5087891
27: -101.6978149, 66.9344254, -101.6067581, 66.8701248, -168.5679321, 168.5411682
28: -80.4911041, 68.0236206, -80.4101028, 67.9745789, -148.4656830, 148.4337158
29: -107.4442825, 58.5641594, -107.2794876, 58.5197563, -165.9640198, 165.8436432
30: -101.2694321, 74.4992828, -101.1334534, 74.4625397, -175.7319641, 175.6327209
31: -107.5201874, 67.2984924, -107.2686768, 67.2655029, -174.7856750, 174.5671692
32: -105.8478317, 69.6469879, -105.6495590, 69.6051941, -175.4530029, 175.2965393
33: -139.9125366, 92.5603485, -139.8531342, 92.3446884, -232.2572021, 232.4134827
34: -118.9944687, 64.3412781, -118.9356918, 64.1702881, -183.1647644, 183.2769775
35: -116.0499115, 71.9837952, -116.0146484, 71.7767792, -187.8266907, 187.9984436
36: -113.8853836, 71.8474884, -113.8237915, 71.7569427, -185.6423340, 185.6712799
37: -167.7272339, 74.4949799, -167.5831299, 74.4333038, -242.1605225, 242.0781097
38: -134.1767273, 83.5640106, -134.0809326, 83.4279022, -217.6046295, 217.6449280
39: -157.8665771, 87.8437195, -157.7069397, 87.7390289, -245.6056061, 245.5506592
40: -124.5457382, 72.1400223, -124.4194946, 72.0650482, -196.6107788, 196.5595093
41: -112.3470993, 81.2158127, -112.2564087, 81.1561966, -193.5032959, 193.4722290
42: -79.5967026, 72.6149292, -79.4040909, 72.5708618, -152.1675568, 152.0190125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=604, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.7626060, upper bound: 102.7690134
time: 145.85 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.7626060, upper bound: 102.8831545
time: 394.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 542.86 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 542.86
Output dim: 5, lower bound: -102.7626060, upper bound: 102.7689310
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 542.86
Output dim: 5, lower bound: -102.7626060, upper bound: 102.8826144
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 542.86
Output dim: 5, lower bound: -102.7626060, upper bound: 102.7690134
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 542.86
Output dim: 5, lower bound: -102.7626060, upper bound: 102.8831545

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -131.8210144, 88.8755341, -131.8486481, 88.8841400, -220.7051239, 220.7241821
1: -68.0530472, 68.4902649, -68.0823212, 68.4961700, -136.5492249, 136.5725708
2: -61.2388382, 70.9263306, -61.2798691, 70.9283752, -132.1672058, 132.2062073
3: -69.9746246, 83.4850464, -70.0270767, 83.4979553, -153.4725494, 153.5121155
4: -76.2938843, 82.2709961, -76.3441925, 82.2783051, -158.5721741, 158.6151886
5: -67.9042969, 85.7616119, -67.9615936, 85.7757568, -153.6800537, 153.7232056
6: -108.1409760, 80.6971741, -108.1664124, 80.7055511, -188.8465118, 188.8635712
7: -79.9389725, 80.9567871, -79.9954987, 80.9636383, -160.9025879, 160.9522705
8: -89.4075928, 102.1556931, -89.4785614, 102.1606598, -191.5682526, 191.6342468
9: -75.5651550, 80.3154907, -75.5356216, 80.4286652, -155.9938202, 155.8511047
10: -110.9488144, 104.4469833, -110.9673538, 104.6424332, -215.5912476, 215.4143372
11: -105.1446075, 64.4842682, -105.1550598, 64.4829102, -169.6275177, 169.6393280
12: -110.3833160, 85.4623566, -110.3895187, 85.6159821, -195.9992981, 195.8518677
13: -108.3084030, 109.5601044, -108.2109528, 109.5848389, -217.8932495, 217.7710419
14: -167.4445343, 95.0237350, -167.4525604, 95.0655365, -262.5100708, 262.4762878
15: -88.2537766, 76.2740021, -88.2650375, 76.2914124, -164.5451965, 164.5390320
16: -111.4562149, 80.0732498, -111.4662933, 80.1413727, -191.5975952, 191.5395508
17: -159.4137115, 83.7702408, -159.3985901, 83.7461853, -243.1598969, 243.1688232
18: -107.3642120, 79.2283020, -107.3773270, 79.1795654, -186.5437469, 186.6056213
19: -80.8932495, 48.6582642, -80.9069977, 48.6128922, -129.5061340, 129.5652618
20: -76.8296585, 60.7423935, -76.8438263, 60.7190361, -137.5486908, 137.5862122
21: -100.4394684, 60.2486153, -100.4471359, 60.2093201, -160.6487885, 160.6957550
22: -102.9153290, 62.1113281, -102.9240723, 62.0895653, -165.0048828, 165.0354004
23: -82.0795059, 61.6582718, -82.0905609, 61.6004715, -143.6799774, 143.7488251
24: -101.4433517, 63.1105576, -101.4686584, 63.0243988, -164.4677429, 164.5792236
25: -87.1739120, 66.0765305, -87.1776657, 66.0489197, -153.2228394, 153.2541809
26: -116.7622147, 94.1355286, -116.7714081, 94.1401291, -210.9023438, 210.9069214
27: -101.4418259, 66.8258362, -101.4977875, 66.7282181, -168.1700439, 168.3236237
28: -80.3196564, 67.9175873, -80.3294983, 67.8397980, -148.1594543, 148.2470856
29: -107.2020645, 58.3122139, -107.2037582, 58.2780838, -165.4801483, 165.5159607
30: -101.0540924, 74.2903214, -101.0594101, 74.2209320, -175.2750244, 175.3497314
31: -107.1467133, 67.1315155, -107.1693344, 67.0939941, -174.2407074, 174.3008423
32: -105.5623779, 69.4229431, -105.5661163, 69.4699402, -175.0323181, 174.9890594
33: -139.6206207, 92.2541733, -139.6330872, 92.2633209, -231.8839111, 231.8872681
34: -118.7801971, 64.0901184, -118.8317490, 64.0908203, -182.8710175, 182.9218445
35: -115.8215866, 71.7150040, -115.8882751, 71.7270508, -187.5486298, 187.6032715
36: -113.7192917, 71.6921082, -113.7497177, 71.6777191, -185.3970032, 185.4418182
37: -167.4634857, 74.2872086, -167.4865723, 74.2832947, -241.7467651, 241.7737732
38: -133.9121704, 83.3564224, -133.9593506, 83.3583145, -217.2704773, 217.3157654
39: -157.5838013, 87.6679153, -157.5853882, 87.6804657, -245.2642670, 245.2532959
40: -124.2903137, 72.0222626, -124.3142242, 72.0232162, -196.3134918, 196.3364563
41: -112.1514587, 81.0807648, -112.1713028, 81.0705414, -193.2220001, 193.2520752
42: -79.3160095, 72.4077301, -79.3316803, 72.4538879, -151.7698975, 151.7394104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.6422241, upper bound: 102.7689310
time: 160.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.6422241, upper bound: 102.7689310
time: 184.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -131.7997742, 88.8678894, -132.1066589, 89.2147369, -221.0145111, 220.9745483
1: -68.0382538, 68.4824982, -68.2549896, 68.7835464, -136.8217926, 136.7374878
2: -61.2191544, 70.9196472, -61.4933434, 71.3417816, -132.5609131, 132.4129791
3: -69.9525299, 83.4769287, -70.2707672, 83.9837723, -153.9362946, 153.7476807
4: -76.2734070, 82.2635498, -76.5727692, 82.6606750, -158.9340820, 158.8363190
5: -67.8846741, 85.7538757, -68.1896515, 86.2988739, -154.1835480, 153.9435272
6: -108.1327667, 80.6599426, -108.3637543, 80.7829056, -188.9156494, 189.0236816
7: -79.9208221, 80.9430008, -80.2470245, 81.3150024, -161.2358246, 161.1900330
8: -89.3911438, 102.1453552, -89.6822968, 102.5540771, -191.9452209, 191.8276520
9: -75.5473175, 80.3033752, -75.6702423, 80.5962372, -156.1435547, 155.9736023
10: -110.9384003, 104.4154816, -111.3203430, 104.9513092, -215.8897095, 215.7358246
11: -105.1324463, 64.4566956, -105.8183746, 64.7443695, -169.8768005, 170.2750549
12: -110.3701096, 85.4322662, -111.0478210, 85.9436722, -196.3137817, 196.4800873
13: -108.2570648, 109.5496826, -108.3511887, 110.0014267, -218.2584839, 217.9008636
14: -167.4239044, 94.9996719, -168.0121155, 95.3461914, -262.7700806, 263.0117798
15: -88.2084122, 76.2649384, -88.4439392, 76.5508881, -164.7593079, 164.7088623
16: -111.4423370, 80.0176010, -111.7971268, 80.2517471, -191.6940918, 191.8147278
17: -159.3955688, 83.7449112, -160.1475677, 84.0326691, -243.4282379, 243.8924866
18: -107.3525848, 79.2047958, -107.9270477, 79.4225082, -186.7750854, 187.1318359
19: -80.8853912, 48.6448822, -81.3589325, 48.7575836, -129.6429749, 130.0038147
20: -76.8216400, 60.7286377, -77.1487656, 60.8566780, -137.6783142, 137.8774109
21: -100.4303055, 60.2291641, -101.0182800, 60.4179115, -160.8482208, 161.2474365
22: -102.8962402, 62.0958748, -103.2780838, 62.2894020, -165.1856384, 165.3739624
23: -82.0710297, 61.6435089, -82.4802628, 61.7615242, -143.8325500, 144.1237793
24: -101.4313202, 63.0976944, -101.7842331, 63.1621246, -164.5934296, 164.8819275
25: -87.1613007, 66.0625305, -87.4223557, 66.2075348, -153.3688354, 153.4848938
26: -116.7432861, 94.1032257, -117.4732590, 94.4927063, -211.2359924, 211.5764771
27: -101.4315796, 66.8071136, -101.8702774, 66.8593979, -168.2909851, 168.6773834
28: -80.3116302, 67.9045105, -80.6816254, 67.9840393, -148.2956696, 148.5861359
29: -107.1850891, 58.2942352, -107.6537933, 58.4872131, -165.6722717, 165.9480286
30: -101.0433960, 74.2652130, -101.4802704, 74.4511719, -175.4945679, 175.7454834
31: -107.1369171, 67.1162262, -107.6648712, 67.2516632, -174.3885651, 174.7810974
32: -105.5527649, 69.4005585, -105.7458801, 69.5910797, -175.1438446, 175.1464233
33: -139.5997772, 92.2452087, -139.8569946, 92.5832062, -232.1829681, 232.1022034
34: -118.7672806, 64.0775375, -118.9786224, 64.2651138, -183.0323944, 183.0561523
35: -115.7929077, 71.7103729, -115.9896698, 71.8946991, -187.6876068, 187.7000427
36: -113.7036438, 71.6836777, -113.8899307, 71.8144531, -185.5180969, 185.5736084
37: -167.4491882, 74.2597504, -167.8247681, 74.4290771, -241.8782349, 242.0845032
38: -133.8937836, 83.3396912, -134.1715088, 83.5094528, -217.4031982, 217.5112000
39: -157.5662537, 87.6601334, -157.7731323, 87.9269485, -245.4931946, 245.4332581
40: -124.2792664, 72.0031433, -124.5364075, 72.1412201, -196.4204865, 196.5395508
41: -112.1435852, 81.0482407, -112.3544464, 81.1618729, -193.3054504, 193.4026794
42: -79.3088684, 72.3728180, -79.5198669, 72.5881195, -151.8969879, 151.8926697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=794, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.6711031, upper bound: 102.8748351
time: 472.54 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -102.6711031, upper bound: 102.8745504
time: 2094.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2569.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2569.65
Output dim: 5, lower bound: -102.6422241, upper bound: 102.7689310
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2569.65
Output dim: 5, lower bound: -102.6422241, upper bound: 102.7689310
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2569.65
Output dim: 5, lower bound: -102.6711031, upper bound: 102.8748351
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2569.65
Output dim: 5, lower bound: -102.6711031, upper bound: 102.8745504
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2569.65
Output dim: 5, lower bound: -102.7626060, upper bound: 102.7690134
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2569.65
Output dim: 5, lower bound: -102.7626060, upper bound: 102.8831545
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=154.09011840820312
rel_dist={5: [-102.90009993993365, 102.90009993642099]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0759516, upper bound: 98.1409406
time: 163.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0759516, upper bound: 98.1484935
time: 140.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 304.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 304.60
Output dim: 5, lower bound: -98.0759516, upper bound: 98.1409406
IS_A2, status: Status.UNKNOWN, split count: 1, time: 304.60
Output dim: 5, lower bound: -98.0759516, upper bound: 98.1484935

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -131.8210144, 88.8755341, -131.9392090, 88.9060287, -220.7270508, 220.8147430
1: -68.0530472, 68.4902649, -68.1452179, 68.5121002, -136.5651550, 136.6354675
2: -61.2388382, 70.9263306, -61.3783226, 70.9475250, -132.1863556, 132.3046570
3: -69.9746246, 83.4850464, -70.1391907, 83.5187225, -153.4933472, 153.6242371
4: -76.2938843, 82.2709961, -76.4434662, 82.2989655, -158.5928345, 158.7144623
5: -67.9042969, 85.7616119, -68.0561066, 85.7945709, -153.6988525, 153.8177185
6: -108.1409760, 80.6971741, -108.1930389, 80.7488098, -188.8897705, 188.8901978
7: -79.9389725, 80.9567871, -80.0738068, 80.9819489, -160.9209290, 161.0305939
8: -89.4075928, 102.1556931, -89.5474701, 102.1864471, -191.5940399, 191.7031555
9: -75.5651550, 80.3154907, -75.6060104, 80.4249954, -155.9901428, 155.9214935
10: -110.9488144, 104.4469833, -111.0065918, 104.6940308, -215.6428528, 215.4535828
11: -105.1446075, 64.4842682, -105.1910095, 64.6382141, -169.7828064, 169.6752625
12: -110.3833160, 85.4623566, -110.4225845, 85.7316360, -196.1149597, 195.8849182
13: -108.3084030, 109.5601044, -108.3575745, 109.6230469, -217.9314270, 217.9176636
14: -167.4445343, 95.0237350, -167.5176697, 95.2101288, -262.6546631, 262.5414124
15: -88.2537766, 76.2740021, -88.3644714, 76.3205261, -164.5743103, 164.6384735
16: -111.4562149, 80.0732498, -111.5203476, 80.1840210, -191.6402283, 191.5935974
17: -159.4137115, 83.7702408, -159.4595795, 83.9160080, -243.3297119, 243.2298279
18: -107.3642120, 79.2283020, -107.4134903, 79.3343124, -186.6985168, 186.6417847
19: -80.8932495, 48.6582642, -80.9357605, 48.7128601, -129.6061096, 129.5940247
20: -76.8296585, 60.7423935, -76.8740082, 60.8037376, -137.6333923, 137.6163940
21: -100.4394684, 60.2486153, -100.4834442, 60.3452682, -160.7847290, 160.7320557
22: -102.9153290, 62.1113281, -102.9605865, 62.1899376, -165.1052704, 165.0718994
23: -82.0795059, 61.6582718, -82.1175537, 61.7076187, -143.7871246, 143.7758179
24: -101.4433517, 63.1105576, -101.4917603, 63.1365547, -164.5799103, 164.6023254
25: -87.1739120, 66.0765305, -87.2074966, 66.1427765, -153.3166809, 153.2840271
26: -116.7622147, 94.1355286, -116.8122559, 94.3365173, -211.0987244, 210.9477692
27: -101.4418259, 66.8258362, -101.5204086, 66.8477249, -168.2895508, 168.3462524
28: -80.3196564, 67.9175873, -80.3624268, 67.9460831, -148.2657471, 148.2800140
29: -107.2020645, 58.3122139, -107.2410965, 58.4081841, -165.6102448, 165.5533142
30: -101.0540924, 74.2903214, -101.0924301, 74.3748016, -175.4288635, 175.3827515
31: -107.1467133, 67.1315155, -107.2042236, 67.1960449, -174.3427429, 174.3357391
32: -105.5623779, 69.4229431, -105.6044998, 69.5069733, -175.0693512, 175.0274353
33: -139.6206207, 92.2541733, -139.7299042, 92.2973938, -231.9179993, 231.9840698
34: -118.7801971, 64.0901184, -118.8535385, 64.1283188, -182.9085083, 182.9436646
35: -115.8215866, 71.7150040, -115.9129562, 71.7440033, -187.5655823, 187.6279602
36: -113.7192917, 71.6921082, -113.7710571, 71.7229614, -185.4422607, 185.4631653
37: -167.4634857, 74.2872086, -167.5214386, 74.3569031, -241.8203583, 241.8086548
38: -133.9121704, 83.3564224, -133.9943237, 83.3905869, -217.3027344, 217.3507385
39: -157.5838013, 87.6679153, -157.6429443, 87.7034912, -245.2872925, 245.3108521
40: -124.2903137, 72.0222626, -124.3517914, 72.0460815, -196.3363953, 196.3740387
41: -112.1514587, 81.0807648, -112.2015991, 81.1198425, -193.2713013, 193.2823486
42: -79.3160095, 72.4077301, -79.3583984, 72.4963989, -151.8124084, 151.7661285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=604, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0657578, upper bound: 98.0715968
time: 554.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0657578, upper bound: 98.1303956
time: 159.26 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -132.1593323, 89.2062988, -132.0600281, 88.9365692, -221.0959015, 221.2663116
1: -68.2940826, 68.7705917, -68.2407074, 68.5325470, -136.8266296, 137.0112915
2: -61.5778618, 71.2851944, -61.5267754, 70.9667816, -132.5446167, 132.8119659
3: -70.3527451, 83.9127960, -70.3105469, 83.5499268, -153.9026642, 154.2233429
4: -76.6697540, 82.6348877, -76.6037827, 82.3237762, -158.9935150, 159.2386627
5: -68.2892303, 86.2218170, -68.2257156, 85.8261108, -154.1153259, 154.4475403
6: -108.3680878, 80.8438950, -108.2429199, 80.7747650, -189.1428528, 189.0868073
7: -80.3261566, 81.3207169, -80.2229233, 81.0012360, -161.3273926, 161.5436401
8: -89.7581329, 102.5694275, -89.6968689, 102.2153931, -191.9735260, 192.2662964
9: -75.8355179, 80.6151505, -75.6453247, 80.5400543, -156.3755798, 156.2604675
10: -111.6241302, 105.0287094, -111.0648804, 104.9552460, -216.5793610, 216.0935974
11: -105.6300659, 64.8383026, -105.2352142, 64.7982254, -170.4282837, 170.0735016
12: -111.1356125, 86.0920715, -110.4572678, 86.0210495, -197.1566467, 196.5493164
13: -108.4429474, 109.8420029, -108.3841705, 109.6872940, -218.1302490, 218.2261658
14: -168.0380554, 95.4540558, -167.5857849, 95.4100647, -263.4480896, 263.0398254
15: -88.5243835, 76.5226440, -88.4388733, 76.3663483, -164.8907318, 164.9615173
16: -111.8295441, 80.3562775, -111.5819473, 80.2719727, -192.1015167, 191.9382019
17: -159.9101410, 84.1549072, -159.5003510, 84.0761566, -243.9862671, 243.6552582
18: -107.7147980, 79.4974518, -107.4619598, 79.4423218, -187.1571198, 186.9594116
19: -81.2070465, 48.7980042, -80.9786835, 48.7686653, -129.9757080, 129.7766876
20: -77.1103516, 60.8931122, -76.9195557, 60.8659401, -137.9762878, 137.8126678
21: -100.8555298, 60.4888954, -100.5260773, 60.4475021, -161.3029938, 161.0149689
22: -103.1307068, 62.3346062, -102.9986725, 62.2552681, -165.3859711, 165.3332825
23: -82.3119354, 61.7857437, -82.1549606, 61.7456398, -144.0575714, 143.9407043
24: -101.6200409, 63.1723137, -101.5329590, 63.1480179, -164.7680359, 164.7052765
25: -87.3523483, 66.2720261, -87.2372513, 66.2129059, -153.5652466, 153.5092621
26: -117.3033447, 94.6449585, -116.8564072, 94.5597000, -211.8630371, 211.5013428
27: -101.6978149, 66.9344254, -101.5976791, 66.8664017, -168.5642090, 168.5321045
28: -80.4911041, 68.0236206, -80.4059601, 67.9695892, -148.4606934, 148.4295807
29: -107.4442825, 58.5641594, -107.2722626, 58.5114708, -165.9557495, 165.8364258
30: -101.2694321, 74.4992828, -101.1284943, 74.4465027, -175.7159424, 175.6277618
31: -107.5201874, 67.2984924, -107.2631912, 67.2581787, -174.7783661, 174.5616760
32: -105.8478317, 69.6469879, -105.6440964, 69.5985565, -175.4463806, 175.2910767
33: -139.9125366, 92.5603485, -139.8432770, 92.3398132, -232.2523193, 232.4036255
34: -118.9944687, 64.3412781, -118.9285965, 64.1658707, -183.1603394, 183.2698669
35: -116.0499115, 71.9837952, -116.0055008, 71.7740707, -187.8239746, 187.9892883
36: -113.8853836, 71.8474884, -113.8148651, 71.7531586, -185.6385498, 185.6623535
37: -167.7272339, 74.4949799, -167.5748291, 74.4236145, -242.1508179, 242.0698090
38: -134.1767273, 83.5640106, -134.0689697, 83.4238281, -217.6005554, 217.6329498
39: -157.8665771, 87.8437195, -157.6995544, 87.7325287, -245.5991058, 245.5432739
40: -124.5457382, 72.1400223, -124.4128189, 72.0573425, -196.6030884, 196.5528259
41: -112.3470993, 81.2158127, -112.2506256, 81.1472931, -193.4943848, 193.4664154
42: -79.5967026, 72.6149292, -79.3988342, 72.5466919, -152.1433716, 152.0137634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=604, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0657578, upper bound: 98.0785668
time: 138.42 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0657578, upper bound: 98.1371653
time: 171.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 312.70 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 312.70
Output dim: 5, lower bound: -98.0657578, upper bound: 98.0715968
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 312.70
Output dim: 5, lower bound: -98.0657578, upper bound: 98.1303956
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 312.70
Output dim: 5, lower bound: -98.0657578, upper bound: 98.0785668
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 312.70
Output dim: 5, lower bound: -98.0657578, upper bound: 98.1371653

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -131.7467346, 88.8571320, -131.7622375, 88.8620605, -220.6087799, 220.6193237
1: -67.9986420, 68.4768982, -68.0155792, 68.4802551, -136.4788818, 136.4924622
2: -61.1552963, 70.9118042, -61.1790848, 70.9129333, -132.0682373, 132.0908813
3: -69.8780670, 83.4659576, -69.9085999, 83.4733200, -153.3513794, 153.3745575
4: -76.2065125, 82.2537842, -76.2364502, 82.2579193, -158.4644318, 158.4902344
5: -67.8180389, 85.7436523, -67.8502197, 85.7518539, -153.5698853, 153.5938721
6: -108.1138077, 80.6631470, -108.1282806, 80.6680374, -188.7817993, 188.7914276
7: -79.8641586, 80.9413300, -79.8948822, 80.9451447, -160.8092957, 160.8362122
8: -89.3363266, 102.1354980, -89.3777924, 102.1382904, -191.4746094, 191.5132904
9: -75.5232239, 80.2837524, -75.5059967, 80.3493347, -155.8725586, 155.7897491
10: -110.9146576, 104.3510590, -110.9252701, 104.4649124, -215.3795624, 215.2763367
11: -105.1151352, 64.3728027, -105.1209106, 64.3718262, -169.4869690, 169.4936981
12: -110.3574677, 85.3326416, -110.3608932, 85.4222488, -195.7797241, 195.6935425
13: -108.2319031, 109.5246429, -108.1751556, 109.5387421, -217.7706451, 217.6997986
14: -167.3948822, 94.9072952, -167.3993835, 94.9316559, -262.3265381, 262.3066711
15: -88.1780853, 76.2473755, -88.1840744, 76.2570038, -164.4350739, 164.4314423
16: -111.4138870, 80.0222015, -111.4194641, 80.0616608, -191.4755249, 191.4416656
17: -159.3740997, 83.6551743, -159.3650818, 83.6408997, -243.0149994, 243.0202637
18: -107.3339539, 79.1310349, -107.3411331, 79.1032562, -186.4371948, 186.4721680
19: -80.8681793, 48.5998611, -80.8760834, 48.5734367, -129.4416199, 129.4759369
20: -76.8035126, 60.6883736, -76.8116608, 60.6748047, -137.4783020, 137.5000305
21: -100.4107513, 60.1625099, -100.4149933, 60.1397057, -160.5504456, 160.5774994
22: -102.8861694, 62.0449638, -102.8909836, 62.0313683, -164.9175415, 164.9359436
23: -82.0564575, 61.5984154, -82.0627823, 61.5647964, -143.6212463, 143.6611786
24: -101.4185791, 63.0558128, -101.4328156, 63.0055161, -164.4240723, 164.4886017
25: -87.1510925, 66.0169830, -87.1532593, 66.0003510, -153.1514435, 153.1702423
26: -116.7297211, 93.9929123, -116.7348557, 93.9954987, -210.7251892, 210.7277679
27: -101.4087524, 66.7691650, -101.4403687, 66.7122955, -168.1210480, 168.2095337
28: -80.2928162, 67.8643188, -80.2984924, 67.8189240, -148.1117249, 148.1628113
29: -107.1743469, 58.2283669, -107.1750336, 58.2074814, -165.3818054, 165.4033966
30: -101.0284653, 74.1996155, -101.0314331, 74.1582947, -175.1867676, 175.2310333
31: -107.1144714, 67.0692978, -107.1275330, 67.0474396, -174.1618958, 174.1968384
32: -105.5335312, 69.3813858, -105.5355225, 69.4082108, -174.9417419, 174.9169006
33: -139.5470734, 92.2266159, -139.5543518, 92.2317352, -231.7787628, 231.7809753
34: -118.7488785, 64.0625610, -118.7788696, 64.0628510, -182.8117218, 182.8414307
35: -115.7835617, 71.6989670, -115.8224335, 71.7058258, -187.4893494, 187.5213928
36: -113.6944275, 71.6634216, -113.7117157, 71.6551971, -185.3496094, 185.3751373
37: -167.4308777, 74.2347260, -167.4440002, 74.2314301, -241.6623077, 241.6787262
38: -133.8725128, 83.3324814, -133.8998413, 83.3334656, -217.2059784, 217.2323151
39: -157.5415955, 87.6474075, -157.5424194, 87.6545181, -245.1961060, 245.1898193
40: -124.2558823, 72.0051575, -124.2696533, 72.0055618, -196.2614441, 196.2747803
41: -112.1234436, 81.0480881, -112.1348877, 81.0420990, -193.1655426, 193.1829834
42: -79.2918549, 72.3630600, -79.3007965, 72.3892670, -151.6811218, 151.6638489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0581354
time: 150.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0581354
time: 194.43 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -131.7867737, 88.8632889, -132.0202637, 89.1928711, -220.9796448, 220.8835449
1: -68.0293503, 68.4776764, -68.1883011, 68.7677612, -136.7971191, 136.6659851
2: -61.2072678, 70.9154968, -61.3925323, 71.3264771, -132.5337219, 132.3080292
3: -69.9389801, 83.4720078, -70.1522827, 83.9593506, -153.8983307, 153.6242676
4: -76.2612305, 82.2590027, -76.4650574, 82.6405487, -158.9017792, 158.7240448
5: -67.8726807, 85.7492294, -68.0781937, 86.2752075, -154.1478882, 153.8274231
6: -108.1277771, 80.6361771, -108.3256836, 80.7452698, -188.8730469, 188.9618530
7: -79.9098511, 80.9344635, -80.1465073, 81.2966537, -161.2065125, 161.0809631
8: -89.3814545, 102.1388702, -89.5815887, 102.5318985, -191.9133606, 191.7204590
9: -75.5365906, 80.2963486, -75.6405029, 80.5175400, -156.0541382, 155.9368591
10: -110.9321594, 104.3961868, -111.2783508, 104.7735596, -215.7057037, 215.6745300
11: -105.1253128, 64.4397888, -105.7845612, 64.6332550, -169.7585754, 170.2243500
12: -110.3619156, 85.4137650, -111.0193939, 85.7499619, -196.1118622, 196.4331665
13: -108.2260590, 109.5432510, -108.3152237, 109.9561386, -218.1821899, 217.8584747
14: -167.4112854, 94.9843140, -167.9594116, 95.2123337, -262.6235962, 262.9437256
15: -88.1817017, 76.2594299, -88.3627396, 76.5170898, -164.6987610, 164.6221619
16: -111.4338913, 79.9835205, -111.7504349, 80.1717072, -191.6055756, 191.7339478
17: -159.3844299, 83.7289429, -160.1142273, 83.9275360, -243.3119354, 243.8431549
18: -107.3453903, 79.1899872, -107.8914871, 79.3460693, -186.6914673, 187.0814819
19: -80.8806000, 48.6365585, -81.3281860, 48.7181702, -129.5987701, 129.9647522
20: -76.8167877, 60.7200813, -77.1168060, 60.8124161, -137.6292114, 137.8368835
21: -100.4246368, 60.2170067, -100.9864197, 60.3483467, -160.7729797, 161.2034302
22: -102.8843994, 62.0862389, -103.2454376, 62.2321739, -165.1165771, 165.3316803
23: -82.0658493, 61.6343803, -82.4527130, 61.7257538, -143.7915955, 144.0870972
24: -101.4241028, 63.0894775, -101.7491150, 63.1432571, -164.5673523, 164.8385925
25: -87.1535339, 66.0539398, -87.3981018, 66.1589813, -153.3125153, 153.4520264
26: -116.7317123, 94.0827637, -117.4369812, 94.3478928, -211.0795898, 211.5197449
27: -101.4254150, 66.7968292, -101.8157806, 66.8432999, -168.2687073, 168.6126099
28: -80.3067169, 67.8963776, -80.6508942, 67.9629974, -148.2697144, 148.5472717
29: -107.1746063, 58.2830200, -107.6253815, 58.4167175, -165.5913239, 165.9083862
30: -101.0369568, 74.2496338, -101.4525757, 74.3884735, -175.4254150, 175.7022095
31: -107.1308899, 67.1067505, -107.6234436, 67.2050934, -174.3359680, 174.7301636
32: -105.5468216, 69.3867035, -105.7154160, 69.5291443, -175.0759583, 175.1021118
33: -139.5867767, 92.2399750, -139.7783508, 92.5517502, -232.1385193, 232.0183258
34: -118.7591248, 64.0697327, -118.9256821, 64.2368011, -182.9959259, 182.9954224
35: -115.7751999, 71.7076492, -115.9236145, 71.8733826, -187.6485596, 187.6312561
36: -113.6939316, 71.6784363, -113.8520660, 71.7915802, -185.4855042, 185.5305023
37: -167.4405212, 74.2449112, -167.7825317, 74.3772278, -241.8177490, 242.0274353
38: -133.8826294, 83.3290405, -134.1121826, 83.4846725, -217.3673096, 217.4412231
39: -157.5556030, 87.6554108, -157.7301483, 87.9012833, -245.4568787, 245.3855591
40: -124.2725220, 71.9913406, -124.4920197, 72.1237946, -196.3963013, 196.4833679
41: -112.1387558, 81.0273514, -112.3180695, 81.1333237, -193.2720642, 193.3453979
42: -79.3044739, 72.3519363, -79.4892273, 72.5239944, -151.8284607, 151.8411560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=794, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
time: 150.95 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
time: 156.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -132.0849609, 89.1880646, -131.8828735, 88.8926315, -220.9775696, 221.0709381
1: -68.2396240, 68.7573395, -68.1109467, 68.5007401, -136.7403564, 136.8682861
2: -61.4942398, 71.2707977, -61.3273582, 70.9322281, -132.4264526, 132.5981598
3: -70.2560272, 83.8938065, -70.0798035, 83.5045624, -153.7605896, 153.9736023
4: -76.5823669, 82.6179581, -76.3964539, 82.2828827, -158.8652496, 159.0144043
5: -68.2028656, 86.2039566, -68.0196686, 85.7834473, -153.9862976, 154.2236176
6: -108.3409424, 80.8097839, -108.1781769, 80.6942444, -189.0351868, 188.9879608
7: -80.2510071, 81.3054047, -80.0435486, 80.9644470, -161.2154541, 161.3489532
8: -89.6868744, 102.5493927, -89.5270691, 102.1673355, -191.8542023, 192.0764618
9: -75.7936707, 80.5830078, -75.5453568, 80.4640961, -156.2577667, 156.1283569
10: -111.5902405, 104.9325027, -110.9834290, 104.7251663, -216.3153839, 215.9158936
11: -105.6013260, 64.7260590, -105.1652756, 64.5305328, -170.1318665, 169.8913269
12: -111.1099014, 85.9623566, -110.3956451, 85.7115784, -196.8214722, 196.3579865
13: -108.3664474, 109.8066025, -108.2017822, 109.6037827, -217.9702301, 218.0083923
14: -167.9887085, 95.3375244, -167.4677124, 95.1314774, -263.1201782, 262.8052368
15: -88.4480438, 76.4960327, -88.2579498, 76.3031158, -164.7511597, 164.7539825
16: -111.7874908, 80.3045349, -111.4815598, 80.1487274, -191.9362183, 191.7860870
17: -159.8706970, 84.0396881, -159.4057922, 83.8008423, -243.6715393, 243.4454651
18: -107.6849289, 79.3999481, -107.3892059, 79.2108765, -186.8957825, 186.7891541
19: -81.1821289, 48.7395363, -80.9190063, 48.6290779, -129.8112030, 129.6585388
20: -77.0843582, 60.8389969, -76.8572540, 60.7368126, -137.8211670, 137.6962433
21: -100.8270111, 60.4027061, -100.4577637, 60.2417107, -161.0687256, 160.8604736
22: -103.1014633, 62.2678070, -102.9291077, 62.0966682, -165.1981354, 165.1969147
23: -82.2891846, 61.7258835, -82.1003723, 61.6027222, -143.8919067, 143.8262329
24: -101.5951309, 63.1175079, -101.4744186, 63.0170021, -164.6121216, 164.5919189
25: -87.3296509, 66.2123032, -87.1830215, 66.0703125, -153.3999634, 153.3953247
26: -117.2710190, 94.5021515, -116.7790833, 94.2183151, -211.4893188, 211.2812195
27: -101.6639938, 66.8777924, -101.5175476, 66.7308197, -168.3947906, 168.3953400
28: -80.4643631, 67.9704895, -80.3422623, 67.8423767, -148.3067322, 148.3127441
29: -107.4169693, 58.4799156, -107.2062836, 58.3101273, -165.7270966, 165.6861877
30: -101.2439270, 74.4077759, -101.0676117, 74.2286987, -175.4726257, 175.4753723
31: -107.4887085, 67.2360840, -107.1867447, 67.1092377, -174.5979462, 174.4228210
32: -105.8191223, 69.6053085, -105.5752716, 69.4997101, -175.3188171, 175.1805725
33: -139.8388214, 92.5329361, -139.6675415, 92.2742920, -232.1131134, 232.2004700
34: -118.9630585, 64.3139954, -118.8538208, 64.1002960, -183.0633545, 183.1678162
35: -116.0116882, 71.9678650, -115.9147873, 71.7360001, -187.7476807, 187.8826599
36: -113.8603210, 71.8191833, -113.7554474, 71.6852570, -185.5455780, 185.5746307
37: -167.6946411, 74.4421463, -167.4972534, 74.2977600, -241.9924011, 241.9393921
38: -134.1366272, 83.5401993, -133.9742584, 83.3668137, -217.5033875, 217.5144348
39: -157.8244019, 87.8230438, -157.5989685, 87.6835632, -245.5079498, 245.4219971
40: -124.5115051, 72.1229858, -124.3307495, 72.0167694, -196.5282593, 196.4537354
41: -112.3191910, 81.1832123, -112.1840363, 81.0698547, -193.3890381, 193.3672485
42: -79.5734024, 72.5695038, -79.3414917, 72.4393463, -152.0127411, 151.9109955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0676725
time: 137.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0677504
time: 164.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -132.1245117, 89.1939850, -132.1402740, 89.2229004, -221.3474121, 221.3342590
1: -68.2699661, 68.7579651, -68.2832794, 68.7878952, -137.0578461, 137.0412445
2: -61.5457039, 71.2742996, -61.5404816, 71.3454895, -132.8911896, 132.8147736
3: -70.3164673, 83.8996582, -70.3231735, 83.9901276, -154.3065948, 154.2228241
4: -76.6365814, 82.6230011, -76.6246567, 82.6651459, -159.3017273, 159.2476501
5: -68.2570496, 86.2093658, -68.2473602, 86.3063583, -154.5634155, 154.4567261
6: -108.3548889, 80.7826538, -108.3754425, 80.7720108, -189.1268921, 189.1580963
7: -80.2963867, 81.2983246, -80.2943268, 81.3155975, -161.6119843, 161.5926514
8: -89.7316284, 102.5525436, -89.7305374, 102.5605316, -192.2921600, 192.2830811
9: -75.8068085, 80.5957947, -75.6799164, 80.6311111, -156.4379272, 156.2757111
10: -111.6073303, 104.9766617, -111.3360825, 105.0335770, -216.6408844, 216.3127289
11: -105.6111679, 64.7935257, -105.8283844, 64.7915573, -170.4027100, 170.6219177
12: -111.1141052, 86.0425797, -111.0535355, 86.0386353, -197.1527405, 197.0961151
13: -108.3602982, 109.8248672, -108.3420715, 110.0196686, -218.3799744, 218.1669312
14: -168.0046997, 95.4142456, -168.0268250, 95.4118958, -263.4165955, 263.4410706
15: -88.4565430, 76.5080109, -88.4434204, 76.5623856, -165.0189209, 164.9514313
16: -111.8071060, 80.2654037, -111.8121262, 80.2594299, -192.0665283, 192.0774994
17: -159.8808594, 84.1131821, -160.1544495, 84.0869293, -243.9677887, 244.2676086
18: -107.6959534, 79.4585342, -107.9384232, 79.4534302, -187.1493835, 187.3969574
19: -81.1943283, 48.7760773, -81.3707275, 48.7736282, -129.9679565, 130.1468048
20: -77.0973969, 60.8704491, -77.1619110, 60.8741875, -137.9715881, 138.0323486
21: -100.8407135, 60.4569397, -101.0287170, 60.4500504, -161.2907715, 161.4856567
22: -103.0994568, 62.3090019, -103.2826614, 62.2967186, -165.3961792, 165.5916595
23: -82.2982941, 61.7616959, -82.4898758, 61.7638474, -144.0621338, 144.2515564
24: -101.6005096, 63.1511269, -101.7898407, 63.1546173, -164.7551270, 164.9409485
25: -87.3318634, 66.2492294, -87.4275055, 66.2287598, -153.5606232, 153.6767273
26: -117.2726822, 94.5912781, -117.4804840, 94.5703354, -211.8429871, 212.0717621
27: -101.6808624, 66.9022369, -101.8857269, 66.8618164, -168.5426788, 168.7879639
28: -80.4780426, 68.0023499, -80.6941223, 67.9870682, -148.4651184, 148.6964722
29: -107.4167938, 58.5345421, -107.6559830, 58.5188293, -165.9356232, 166.1905212
30: -101.2522430, 74.4584808, -101.4881973, 74.4589386, -175.7111816, 175.9466705
31: -107.5045166, 67.2734985, -107.6819611, 67.2666321, -174.7711487, 174.9554443
32: -105.8321915, 69.6103897, -105.7549057, 69.6204605, -175.4526520, 175.3652954
33: -139.8782959, 92.5460892, -139.8910828, 92.5939026, -232.4721985, 232.4371643
34: -118.9731903, 64.3210144, -119.0005417, 64.2744904, -183.2476807, 183.3215637
35: -116.0030975, 71.9764099, -116.0159912, 71.9036255, -187.9067230, 187.9924011
36: -113.8597107, 71.8338242, -113.8955307, 71.8228149, -185.6825256, 185.7293549
37: -167.7041168, 74.4523926, -167.8349915, 74.4435043, -242.1476135, 242.2873535
38: -134.1467896, 83.5366821, -134.1863098, 83.5176697, -217.6644440, 217.7229767
39: -157.8381653, 87.8311310, -157.7866058, 87.9299469, -245.7681122, 245.6177368
40: -124.5278397, 72.1094894, -124.5525970, 72.1354980, -196.6633301, 196.6620789
41: -112.3343277, 81.1624908, -112.3669891, 81.1616974, -193.4960022, 193.5294800
42: -79.5855179, 72.5587616, -79.5295258, 72.5739517, -152.1594696, 152.0882874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=794, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1270118
time: 186.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1271070
time: 164.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 353.54 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 353.54
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0581354
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 353.54
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0581354
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 353.54
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 353.54
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 353.54
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0676725
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 353.54
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0677504
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 353.54
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1270118
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 353.54
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1271070

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -131.5553741, 88.8027802, -131.6803131, 88.8389206, -220.3942871, 220.4830933
1: -67.8462219, 68.4346924, -67.9506378, 68.4623260, -136.3085480, 136.3853302
2: -60.9478951, 70.8678284, -61.0907440, 70.8942871, -131.8421783, 131.9585724
3: -69.6057663, 83.4012070, -69.7930145, 83.4458160, -153.0515747, 153.1942139
4: -75.9660110, 82.2029190, -76.1339874, 82.2363510, -158.2023621, 158.3369141
5: -67.5904236, 85.6823273, -67.7534485, 85.7258377, -153.3162537, 153.4357758
6: -108.0247650, 80.5933533, -108.0902252, 80.6379395, -188.6626740, 188.6835785
7: -79.6658478, 80.8948975, -79.8102417, 80.9254456, -160.5912781, 160.7051392
8: -89.1458817, 102.0701599, -89.2966461, 102.1106110, -191.2565002, 191.3667908
9: -75.4282074, 80.1796341, -75.4654999, 80.3045349, -155.7327423, 155.6451416
10: -110.8124161, 103.9451218, -110.8816681, 104.2925644, -215.1049805, 214.8267517
11: -105.0319519, 64.0425110, -105.0854340, 64.2316895, -169.2636414, 169.1279297
12: -110.2790070, 84.9694061, -110.3275299, 85.2682343, -195.5472412, 195.2969360
13: -108.0140839, 109.4197540, -108.0828323, 109.4942398, -217.5083313, 217.5025940
14: -167.2481384, 94.5266037, -167.3371429, 94.7707977, -262.0189209, 261.8637390
15: -87.9723129, 76.1670837, -88.0948792, 76.2227325, -164.1950378, 164.2619629
16: -111.2868347, 79.8455811, -111.3652954, 79.9852066, -191.2720337, 191.2108765
17: -159.2714233, 83.3383865, -159.3213501, 83.5069122, -242.7783356, 242.6597290
18: -107.2342529, 78.8555527, -107.2986298, 78.9851303, -186.2193451, 186.1541748
19: -80.7903061, 48.4477234, -80.8430328, 48.5088081, -129.2991180, 129.2907410
20: -76.7237244, 60.5464134, -76.7778625, 60.6146431, -137.3383636, 137.3242798
21: -100.3260574, 59.9308777, -100.3790131, 60.0415573, -160.3676147, 160.3098755
22: -102.8025360, 61.8896217, -102.8554230, 61.9641380, -164.7666626, 164.7450409
23: -81.9868011, 61.4240532, -82.0331802, 61.4887390, -143.4755402, 143.4572144
24: -101.3430176, 62.9032707, -101.4005585, 62.9410248, -164.2840424, 164.3038025
25: -87.0828629, 65.8417511, -87.1243134, 65.9259720, -153.0088196, 152.9660645
26: -116.6335907, 93.6159592, -116.6940308, 93.8358765, -210.4694519, 210.3099670
27: -101.3075714, 66.6706543, -101.3969727, 66.6690979, -167.9766693, 168.0676117
28: -80.2124939, 67.7612762, -80.2644806, 67.7746735, -147.9871674, 148.0257416
29: -107.1008911, 58.0353661, -107.1437836, 58.1253052, -165.2261658, 165.1791534
30: -100.9538574, 73.9449158, -100.9997177, 74.0503387, -175.0041809, 174.9446411
31: -107.0101776, 66.8634491, -107.0832367, 66.9595413, -173.9697113, 173.9466553
32: -105.4340363, 69.2976532, -105.4932251, 69.3723145, -174.8063507, 174.7908783
33: -139.3275452, 92.1552887, -139.4608002, 92.2013092, -231.5288544, 231.6160889
34: -118.6122742, 63.9912567, -118.7208481, 64.0324326, -182.6446686, 182.7120972
35: -115.6085739, 71.6503372, -115.7482910, 71.6850815, -187.2936401, 187.3986206
36: -113.5462570, 71.6122971, -113.6486206, 71.6335526, -185.1798096, 185.2609100
37: -167.3190613, 74.1158218, -167.3963623, 74.1805344, -241.4995728, 241.5121765
38: -133.7042847, 83.2641296, -133.8281555, 83.3043976, -217.0086823, 217.0922852
39: -157.3997192, 87.5878983, -157.4819946, 87.6291656, -245.0288849, 245.0698853
40: -124.1397324, 71.9611053, -124.2203064, 71.9867630, -196.1264496, 196.1814117
41: -112.0224533, 80.9797592, -112.0918732, 81.0128021, -193.0352478, 193.0716248
42: -79.2237244, 72.2382507, -79.2718201, 72.3344803, -151.5581970, 151.5100708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
time: 153.86 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
time: 186.42 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -131.8701324, 89.0559387, -131.7190094, 88.8434296, -220.7135620, 220.7749481
1: -68.0540161, 68.7402496, -67.9818726, 68.4654465, -136.5194702, 136.7221222
2: -61.1946297, 71.2684021, -61.1413078, 70.8982315, -132.0928497, 132.4096985
3: -69.8818207, 83.9755936, -69.8558044, 83.4524689, -153.3342896, 153.8313904
4: -76.2758102, 82.6018829, -76.1947327, 82.2401733, -158.5159912, 158.7966156
5: -67.8549347, 86.2271271, -67.8065262, 85.7320023, -153.5869141, 154.0336456
6: -108.2192078, 80.6762390, -108.1014938, 80.5938416, -188.8130493, 188.7777100
7: -79.9536133, 81.2650299, -79.8545837, 80.9204559, -160.8740692, 161.1196136
8: -89.3827667, 102.4996490, -89.3408737, 102.1172867, -191.5000610, 191.8405151
9: -75.5911407, 80.3634567, -75.4662781, 80.3146667, -155.9058075, 155.8297272
10: -111.5985413, 104.3978806, -110.9002457, 104.3857574, -215.9842987, 215.2981262
11: -105.8132019, 64.3823624, -105.0898666, 64.3154144, -170.1286163, 169.4722290
12: -111.0270386, 85.3578796, -110.3337250, 85.3564835, -196.3835144, 195.6916046
13: -108.1943130, 109.9366989, -108.0746918, 109.5125351, -217.7068481, 218.0113831
14: -168.0677185, 94.8872223, -167.3539429, 94.8693161, -262.9370117, 262.2411499
15: -88.2027512, 76.4437180, -88.0913849, 76.2336655, -164.4364166, 164.5350952
16: -111.7577896, 80.0411835, -111.3832092, 79.9660797, -191.7238770, 191.4243927
17: -160.0959167, 83.6760864, -159.3299866, 83.5847092, -243.6806335, 243.0060730
18: -107.9055557, 79.1459274, -107.3139801, 79.0510864, -186.9566345, 186.4598999
19: -81.3252869, 48.6085892, -80.8564453, 48.5451851, -129.8704681, 129.4650269
20: -77.0816269, 60.7024193, -76.7921753, 60.6452141, -137.7268372, 137.4945984
21: -100.9922714, 60.1759338, -100.3914413, 60.1001205, -161.0923920, 160.5673828
22: -103.1168823, 62.0951500, -102.8539810, 61.9988403, -165.1157074, 164.9491272
23: -82.4372559, 61.6299934, -82.0428085, 61.5301285, -143.9673767, 143.6727905
24: -101.7182922, 63.0255013, -101.4033661, 62.9473381, -164.6656342, 164.4288635
25: -87.4296875, 66.0347214, -87.1256256, 65.9663162, -153.3959961, 153.1603394
26: -117.3702240, 94.0329895, -116.6966400, 93.9288406, -211.2990723, 210.7296295
27: -101.5661621, 66.7361374, -101.4116516, 66.6477661, -168.2139282, 168.1477966
28: -80.4669037, 67.9155884, -80.2793427, 67.7921600, -148.2590637, 148.1949310
29: -107.5136719, 58.2533989, -107.1425629, 58.1724205, -165.6860962, 165.3959351
30: -101.4082794, 74.2395630, -101.0050201, 74.1109467, -175.5192261, 175.2445679
31: -107.6808090, 67.0695801, -107.1036682, 67.0104065, -174.6911926, 174.1732330
32: -105.5952225, 69.4197006, -105.4946594, 69.3658676, -174.9610901, 174.9143524
33: -139.6140442, 92.5772552, -139.5106506, 92.2110443, -231.8250885, 232.0878906
34: -118.7975616, 64.2757721, -118.7424927, 64.0420990, -182.8396606, 183.0182648
35: -115.7982330, 71.9894714, -115.7668076, 71.6938324, -187.4920654, 187.7562866
36: -113.7205353, 71.8559952, -113.6683884, 71.6434555, -185.3639832, 185.5243835
37: -167.6172333, 74.2637100, -167.4063110, 74.1703339, -241.7875366, 241.6700134
38: -133.9685364, 83.5812531, -133.8479919, 83.3121796, -217.2806702, 217.4292450
39: -157.6539917, 87.8447723, -157.4940643, 87.6365891, -245.2905579, 245.3388062
40: -124.3943710, 72.1492691, -124.2394867, 71.9807129, -196.3750763, 196.3887634
41: -112.1893768, 81.1447754, -112.1001434, 81.0069962, -193.1963654, 193.2449188
42: -79.4248657, 72.4157486, -79.2819214, 72.3105545, -151.7354126, 151.6976624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=792, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
time: 317.87 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0080233, upper bound: 98.0550070
time: 140.24 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -131.5945587, 88.8086090, -131.9380035, 89.1697617, -220.7643127, 220.7465973
1: -67.8763351, 68.4351044, -68.1229248, 68.7497559, -136.6260986, 136.5580292
2: -60.9990883, 70.8712540, -61.3037338, 71.3077850, -132.3068542, 132.1749878
3: -69.6659698, 83.4069901, -70.0361938, 83.9318085, -153.5977783, 153.4431763
4: -76.0192795, 82.2079926, -76.3624344, 82.6190491, -158.6383362, 158.5704041
5: -67.6442566, 85.6876984, -67.9808960, 86.2492218, -153.8934631, 153.6685944
6: -108.0384064, 80.5665894, -108.2875748, 80.7146454, -188.7530212, 188.8541565
7: -79.7108765, 80.8876038, -80.0615845, 81.2768707, -160.9877472, 160.9491730
8: -89.1902847, 102.0730667, -89.4999771, 102.5041428, -191.6944275, 191.5730286
9: -75.4413147, 80.1920471, -75.5995865, 80.4725113, -155.9138184, 155.7916260
10: -110.8293457, 103.9882507, -111.2347641, 104.6000671, -215.4294128, 215.2230225
11: -105.0417328, 64.1083832, -105.7491302, 64.4924698, -169.5342102, 169.8574829
12: -110.2831573, 85.0496063, -110.9861450, 85.5952988, -195.8784485, 196.0357513
13: -108.0058212, 109.4380341, -108.2210922, 109.9123306, -217.9181519, 217.6591187
14: -167.2640686, 94.6030960, -167.8972168, 95.0511703, -262.3152466, 262.5003052
15: -87.9743805, 76.1789703, -88.2726212, 76.4833832, -164.4577484, 164.4515991
16: -111.3064499, 79.8021240, -111.6960220, 80.0934601, -191.3998718, 191.4981384
17: -159.2812653, 83.4116211, -160.0704498, 83.7932663, -243.0745239, 243.4820557
18: -107.2447662, 78.9138870, -107.8488922, 79.2275467, -186.4723053, 186.7627716
19: -80.8023300, 48.4838333, -81.2950287, 48.6531487, -129.4554749, 129.7788696
20: -76.7368011, 60.5776939, -77.0829926, 60.7519760, -137.4887695, 137.6606903
21: -100.3397293, 59.9847069, -100.9504471, 60.2498016, -160.5895233, 160.9351501
22: -102.8001099, 61.9304581, -103.2098999, 62.1647034, -164.9647980, 165.1403503
23: -81.9959412, 61.4590340, -82.4232330, 61.6491814, -143.6451263, 143.8822632
24: -101.3481293, 62.9368095, -101.7169037, 63.0786819, -164.4268036, 164.6537170
25: -87.0850143, 65.8781586, -87.3690948, 66.0843048, -153.1693115, 153.2472534
26: -116.6349792, 93.7051544, -117.3960648, 94.1878662, -210.8228455, 211.1012268
27: -101.3237000, 66.6959610, -101.7732544, 66.7991180, -168.1228180, 168.4692078
28: -80.2262421, 67.7926102, -80.6169739, 67.9183044, -148.1445465, 148.4095764
29: -107.1007538, 58.0895157, -107.5941772, 58.3341980, -165.4349213, 165.6836853
30: -100.9620514, 73.9935303, -101.4209595, 74.2796783, -175.2416992, 175.4144897
31: -107.0257874, 66.9003296, -107.5791931, 67.1168976, -174.1426849, 174.4795227
32: -105.4470825, 69.3026810, -105.6731567, 69.4926300, -174.9397125, 174.9758301
33: -139.3666534, 92.1683807, -139.6845703, 92.5211945, -231.8878326, 231.8529358
34: -118.6220703, 63.9980965, -118.8674011, 64.2057419, -182.8277893, 182.8654938
35: -115.5977631, 71.6588593, -115.8480377, 71.8524780, -187.4502258, 187.5068970
36: -113.5455551, 71.6269836, -113.7888641, 71.7695923, -185.3151245, 185.4158478
37: -167.3282166, 74.1236343, -167.7348022, 74.3250809, -241.6532898, 241.8584290
38: -133.7135925, 83.2607269, -134.0398865, 83.4554214, -217.1689758, 217.3006134
39: -157.4137573, 87.5957184, -157.6694641, 87.8759689, -245.2897186, 245.2651825
40: -124.1558151, 71.9475021, -124.4424438, 72.1045685, -196.2603760, 196.3899384
41: -112.0375061, 80.9589310, -112.2749557, 81.1035767, -193.1410675, 193.2338867
42: -79.2362061, 72.2259369, -79.4602203, 72.4690170, -151.7052155, 151.6861420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=794, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
time: 154.14 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
time: 233.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 390.12 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 390.12
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 390.12
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 390.12
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 390.12
Output dim: 5, lower bound: -98.0080233, upper bound: 98.0550070
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 390.12
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 390.12
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 390.12
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 390.12
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0676725
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 390.12
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0677504
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 390.12
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1270118
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 390.12
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1271070
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=154.09011840820312
rel_dist={5: [-98.15387180223486, 98.15387180681547]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0273715, upper bound: 94.0496964
time: 150.06 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0273715, upper bound: 94.0496964
time: 266.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 417.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 417.03
Output dim: 5, lower bound: -94.0273715, upper bound: 94.0496964
IS_A2, status: Status.UNKNOWN, split count: 1, time: 417.03
Output dim: 5, lower bound: -94.0273715, upper bound: 94.0496964

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -131.8210144, 88.8755341, -131.8646088, 88.8869247, -220.7079468, 220.7401428
1: -68.0530472, 68.4902649, -68.0874023, 68.4984055, -136.5514526, 136.5776672
2: -61.2388382, 70.9263306, -61.2909279, 70.9342194, -132.1730499, 132.2172546
3: -69.9746246, 83.4850464, -70.0361557, 83.4975586, -153.4721832, 153.5211792
4: -76.2938843, 82.2709961, -76.3498077, 82.2814331, -158.5753021, 158.6208038
5: -67.9042969, 85.7616119, -67.9610291, 85.7739029, -153.6781921, 153.7226410
6: -108.1409760, 80.6971741, -108.1602783, 80.7164307, -188.8574066, 188.8574524
7: -79.9389725, 80.9567871, -79.9885178, 80.9661636, -160.9051361, 160.9453125
8: -89.4075928, 102.1556931, -89.4598999, 102.1671829, -191.5747681, 191.6155853
9: -75.5651550, 80.3154907, -75.5803528, 80.3563080, -155.9214478, 155.8958435
10: -110.9488144, 104.4469833, -110.9702835, 104.5392990, -215.4881134, 215.4172668
11: -105.1446075, 64.4842682, -105.1618423, 64.5416412, -169.6862488, 169.6461182
12: -110.3833160, 85.4623566, -110.3979416, 85.5631409, -195.9464417, 195.8602753
13: -108.3084030, 109.5601044, -108.3267288, 109.5835419, -217.8919373, 217.8868256
14: -167.4445343, 95.0237350, -167.4718323, 95.0934830, -262.5380249, 262.4955750
15: -88.2537766, 76.2740021, -88.2943573, 76.2912140, -164.5449829, 164.5683594
16: -111.4562149, 80.0732498, -111.4800568, 80.1145935, -191.5708008, 191.5533142
17: -159.4137115, 83.7702408, -159.4308319, 83.8247528, -243.2384644, 243.2010803
18: -107.3642120, 79.2283020, -107.3823853, 79.2678986, -186.6320648, 186.6106720
19: -80.8932495, 48.6582642, -80.9090576, 48.6786270, -129.5718689, 129.5673218
20: -76.8296585, 60.7423935, -76.8462219, 60.7653122, -137.5949402, 137.5886230
21: -100.4394684, 60.2486153, -100.4557877, 60.2847366, -160.7241974, 160.7044067
22: -102.9153290, 62.1113281, -102.9321976, 62.1401749, -165.0554810, 165.0435181
23: -82.0795059, 61.6582718, -82.0936737, 61.6766739, -143.7561798, 143.7519379
24: -101.4433517, 63.1105576, -101.4612350, 63.1202126, -164.5635681, 164.5717926
25: -87.1739120, 66.0765305, -87.1864548, 66.1009369, -153.2748413, 153.2629700
26: -116.7622147, 94.1355286, -116.7808914, 94.2106934, -210.9729004, 210.9163818
27: -101.4418259, 66.8258362, -101.4708862, 66.8339844, -168.2758026, 168.2967224
28: -80.3196564, 67.9175873, -80.3356018, 67.9281540, -148.2478027, 148.2531891
29: -107.2020645, 58.3122139, -107.2165833, 58.3474922, -165.5495605, 165.5287933
30: -101.0540924, 74.2903214, -101.0683823, 74.3218536, -175.3759460, 175.3587036
31: -107.1467133, 67.1315155, -107.1681442, 67.1555939, -174.3023071, 174.2996521
32: -105.5623779, 69.4229431, -105.5780640, 69.4542389, -175.0166016, 175.0010071
33: -139.6206207, 92.2541733, -139.6614532, 92.2702332, -231.8908386, 231.9156189
34: -118.7801971, 64.0901184, -118.8075638, 64.1042786, -182.8844757, 182.8976746
35: -115.8215866, 71.7150040, -115.8557129, 71.7257919, -187.5473633, 187.5707092
36: -113.7192917, 71.6921082, -113.7384033, 71.7035294, -185.4228210, 185.4305115
37: -167.4634857, 74.2872086, -167.4849243, 74.3130722, -241.7765350, 241.7721252
38: -133.9121704, 83.3564224, -133.9427490, 83.3691177, -217.2812500, 217.2991638
39: -157.5838013, 87.6679153, -157.6058044, 87.6811752, -245.2649841, 245.2737122
40: -124.2903137, 72.0222626, -124.3132019, 72.0311279, -196.3214417, 196.3354645
41: -112.1514587, 81.0807648, -112.1701355, 81.0953522, -193.2468109, 193.2509003
42: -79.3160095, 72.4077301, -79.3317871, 72.4404297, -151.7564087, 151.7395020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=604, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=792, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0186277, upper bound: 94.0192999
time: 146.02 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0186277, upper bound: 94.0411577
time: 123.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -132.1593323, 89.2062988, -132.0482483, 88.9337387, -221.0930786, 221.2545471
1: -68.2940826, 68.7705917, -68.2324600, 68.5298920, -136.8239746, 137.0030518
2: -61.5778618, 71.2851944, -61.5158615, 70.9639282, -132.5417938, 132.8010559
3: -70.3527451, 83.9127960, -70.2965240, 83.5455170, -153.8982544, 154.2093201
4: -76.6697540, 82.6348877, -76.5926361, 82.3199387, -158.9896851, 159.2275085
5: -68.2892303, 86.2218170, -68.2152863, 85.8223343, -154.1115570, 154.4371033
6: -108.3680878, 80.8438950, -108.2368164, 80.7592621, -189.1273499, 189.0807037
7: -80.3261566, 81.3207169, -80.2138977, 80.9964142, -161.3225708, 161.5346069
8: -89.7581329, 102.5694275, -89.6863785, 102.2116699, -191.9698029, 192.2557983
9: -75.8355179, 80.6151505, -75.6407471, 80.5308380, -156.3663330, 156.2558899
10: -111.6241302, 105.0287094, -111.0589523, 104.9356766, -216.5598145, 216.0876617
11: -105.6300659, 64.8383026, -105.2299500, 64.7850037, -170.4150696, 170.0682526
12: -111.1356125, 86.0920715, -110.4514618, 86.0015869, -197.1371918, 196.5435181
13: -108.4429474, 109.8420029, -108.3705368, 109.6818237, -218.1247711, 218.2125244
14: -168.0380554, 95.4540558, -167.5766602, 95.3965073, -263.4345703, 263.0307007
15: -88.5243835, 76.5226440, -88.4172211, 76.3613434, -164.8857269, 164.9398651
16: -111.8295441, 80.3562775, -111.5746002, 80.2519836, -192.0815277, 191.9308472
17: -159.9101410, 84.1549072, -159.4936829, 84.0665588, -243.9766693, 243.6485901
18: -107.7147980, 79.4974518, -107.4561234, 79.4324875, -187.1472778, 186.9535522
19: -81.2070465, 48.7980042, -80.9745178, 48.7633972, -129.9704285, 129.7725220
20: -77.1103516, 60.8931122, -76.9157715, 60.8601303, -137.9704742, 137.8088837
21: -100.8555298, 60.4888954, -100.5211411, 60.4398117, -161.2953186, 161.0100403
22: -103.1307068, 62.3346062, -102.9915619, 62.2417755, -165.3724823, 165.3261566
23: -82.3119354, 61.7857437, -82.1509857, 61.7360153, -144.0479431, 143.9367218
24: -101.6200409, 63.1723137, -101.5250168, 63.1397057, -164.7597198, 164.6973267
25: -87.3523483, 66.2720261, -87.2325058, 66.2079773, -153.5603333, 153.5045166
26: -117.3033447, 94.6449585, -116.8493958, 94.5469208, -211.8502655, 211.4943390
27: -101.6978149, 66.9344254, -101.5897064, 66.8628311, -168.5606384, 168.5241394
28: -80.4911041, 68.0236206, -80.4021301, 67.9649277, -148.4560242, 148.4257507
29: -107.4442825, 58.5641594, -107.2656860, 58.5034409, -165.9477081, 165.8298492
30: -101.2694321, 74.4992828, -101.1240082, 74.4314270, -175.7008362, 175.6232910
31: -107.5201874, 67.2984924, -107.2580261, 67.2513428, -174.7715149, 174.5565033
32: -105.8478317, 69.6469879, -105.6391525, 69.5921783, -175.4399872, 175.2861328
33: -139.9125366, 92.5603485, -139.8338013, 92.3353424, -232.2478638, 232.3941498
34: -118.9944687, 64.3412781, -118.9218903, 64.1617050, -183.1561584, 183.2631683
35: -116.0499115, 71.9837952, -115.9968567, 71.7714691, -187.8213806, 187.9806519
36: -113.8853836, 71.8474884, -113.8063812, 71.7495117, -185.6348877, 185.6538696
37: -167.7272339, 74.4949799, -167.5671692, 74.4145889, -242.1418152, 242.0621490
38: -134.1767273, 83.5640106, -134.0574341, 83.4199677, -217.5966797, 217.6214294
39: -157.8665771, 87.8437195, -157.6927795, 87.7263489, -245.5929108, 245.5364990
40: -124.5457382, 72.1400223, -124.4067307, 72.0499268, -196.5956726, 196.5467529
41: -112.3470993, 81.2158127, -112.2453766, 81.1386337, -193.4857330, 193.4611816
42: -79.5967026, 72.6149292, -79.3940430, 72.5237961, -152.1204834, 152.0089722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=604, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0186277, upper bound: 94.0281964
time: 139.12 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0186277, upper bound: 94.0499410
time: 171.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 312.50 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 312.50
Output dim: 5, lower bound: -94.0186277, upper bound: 94.0192999
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 312.50
Output dim: 5, lower bound: -94.0186277, upper bound: 94.0411577
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 312.50
Output dim: 5, lower bound: -94.0186277, upper bound: 94.0281964
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 312.50
Output dim: 5, lower bound: -94.0186277, upper bound: 94.0499410

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -131.6819153, 88.8410110, -131.6878357, 88.8430023, -220.5249176, 220.5288391
1: -67.9511795, 68.4651642, -67.9579468, 68.4665451, -136.4177246, 136.4231110
2: -61.0823021, 70.8990707, -61.0918770, 70.8996124, -131.9819031, 131.9909515
3: -69.7935486, 83.4492493, -69.8058243, 83.4521255, -153.2456665, 153.2550659
4: -76.1311111, 82.2386398, -76.1430817, 82.2403259, -158.3714294, 158.3817139
5: -67.7425461, 85.7279816, -67.7553177, 85.7311707, -153.4737244, 153.4833069
6: -108.0898895, 80.6336365, -108.0954437, 80.6358032, -188.7256927, 188.7290802
7: -79.7986755, 80.9277878, -79.8100586, 80.9293518, -160.7280273, 160.7378540
8: -89.2742081, 102.1177673, -89.2903595, 102.1189957, -191.3931885, 191.4081116
9: -75.4865341, 80.2561035, -75.4804001, 80.2807922, -155.7673187, 155.7364807
10: -110.8847733, 104.2673874, -110.8889160, 104.3108597, -215.1956177, 215.1562805
11: -105.0893173, 64.2751923, -105.0916138, 64.2754822, -169.3647919, 169.3668060
12: -110.3347855, 85.2192078, -110.3362045, 85.2539825, -195.5887756, 195.5554047
13: -108.1649933, 109.4933853, -108.1444016, 109.4988632, -217.6638489, 217.6377563
14: -167.3513489, 94.8049850, -167.3534546, 94.8151703, -262.1665039, 262.1584473
15: -88.1121216, 76.2239075, -88.1142273, 76.2275467, -164.3396606, 164.3381348
16: -111.3768082, 79.9777527, -111.3790512, 79.9929276, -191.3697052, 191.3567963
17: -159.3393555, 83.5541382, -159.3362732, 83.5498581, -242.8891754, 242.8904114
18: -107.3074951, 79.0456696, -107.3103180, 79.0371246, -186.3446198, 186.3559723
19: -80.8462601, 48.5487137, -80.8494110, 48.5393028, -129.3855591, 129.3981171
20: -76.7805634, 60.6410980, -76.7838440, 60.6365128, -137.4170532, 137.4249420
21: -100.3855667, 60.0871010, -100.3873062, 60.0793457, -160.4649048, 160.4744110
22: -102.8606491, 61.9869347, -102.8626785, 61.9825592, -164.8432007, 164.8496094
23: -82.0363617, 61.5460396, -82.0388565, 61.5339813, -143.5703430, 143.5848999
24: -101.3968353, 63.0076180, -101.4022369, 62.9892082, -164.3860474, 164.4098511
25: -87.1311188, 65.9647980, -87.1321640, 65.9588089, -153.0899353, 153.0969543
26: -116.7012558, 93.8677368, -116.7034454, 93.8699493, -210.5711975, 210.5711823
27: -101.3797836, 66.7194061, -101.3916016, 66.6986160, -168.0783997, 168.1109924
28: -80.2693024, 67.8176117, -80.2716217, 67.8010406, -148.0703278, 148.0892334
29: -107.1500244, 58.1548195, -107.1504364, 58.1470566, -165.2970886, 165.3052521
30: -101.0059586, 74.1202545, -101.0072556, 74.1056366, -175.1116028, 175.1275024
31: -107.0862427, 67.0148010, -107.0913925, 67.0071335, -174.0933685, 174.1062012
32: -105.5081635, 69.3452225, -105.5091171, 69.3555679, -174.8637238, 174.8543091
33: -139.4826965, 92.2024689, -139.4861145, 92.2044983, -231.6871948, 231.6885834
34: -118.7215576, 64.0385513, -118.7330322, 64.0388412, -182.7603760, 182.7715759
35: -115.7504654, 71.6849060, -115.7653656, 71.6875458, -187.4379730, 187.4502716
36: -113.6727600, 71.6388168, -113.6792831, 71.6359863, -185.3087158, 185.3181000
37: -167.4024048, 74.1887207, -167.4074402, 74.1878891, -241.5902710, 241.5961456
38: -133.8380432, 83.3115158, -133.8485870, 83.3120041, -217.1500397, 217.1600952
39: -157.5047760, 87.6293030, -157.5054016, 87.6321106, -245.1368408, 245.1347046
40: -124.2257538, 71.9903564, -124.2311401, 71.9906311, -196.2163849, 196.2214966
41: -112.0989227, 81.0196457, -112.1034012, 81.0176544, -193.1165771, 193.1230164
42: -79.2706909, 72.3242493, -79.2742004, 72.3344879, -151.6051636, 151.5984497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=792, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9885538, upper bound: 94.0039467
time: 155.14 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0039467
time: 130.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -131.7728271, 88.8582153, -131.9461212, 89.1739349, -220.9467468, 220.8043213
1: -68.0196762, 68.4723434, -68.1306686, 68.7541428, -136.7738037, 136.6030121
2: -61.1940804, 70.9109192, -61.3053284, 71.3132629, -132.5073242, 132.2162476
3: -69.9238663, 83.4665985, -70.0494843, 83.9383545, -153.8622131, 153.5160828
4: -76.2477722, 82.2540436, -76.3717041, 82.6231537, -158.8709106, 158.6257324
5: -67.8593826, 85.7441559, -67.9832611, 86.2547455, -154.1141357, 153.7274017
6: -108.1222992, 80.6100388, -108.2928848, 80.7129364, -188.8352356, 188.9029236
7: -79.8976593, 80.9247589, -80.0619659, 81.2809906, -161.1786499, 160.9867249
8: -89.3708038, 102.1318130, -89.4941940, 102.5127716, -191.8835754, 191.6260071
9: -75.5254974, 80.2885895, -75.6148376, 80.4495697, -155.9750671, 155.9034271
10: -110.9253540, 104.3747635, -111.2421722, 104.6193542, -215.5447083, 215.6169128
11: -105.1176147, 64.4207764, -105.7555237, 64.5369263, -169.6545410, 170.1763000
12: -110.3526230, 85.3933563, -110.9950027, 85.5816879, -195.9342957, 196.3883667
13: -108.1961594, 109.5363312, -108.2843857, 109.9170380, -218.1131744, 217.8207092
14: -167.3973694, 94.9668427, -167.9138184, 95.0958633, -262.4932251, 262.8806763
15: -88.1516113, 76.2534790, -88.2925873, 76.4881592, -164.6397705, 164.5460510
16: -111.4248581, 79.9446869, -111.7101517, 80.1026535, -191.5275116, 191.6548462
17: -159.3720093, 83.7108154, -160.0856018, 83.8365021, -243.2084961, 243.7963867
18: -107.3376160, 79.1733551, -107.8609619, 79.2798386, -186.6174622, 187.0343170
19: -80.8753128, 48.6271515, -81.3016357, 48.6840553, -129.5593567, 129.9287872
20: -76.8114548, 60.7106400, -77.0890884, 60.7740936, -137.5855408, 137.7997284
21: -100.4183960, 60.2033539, -100.9589386, 60.2879829, -160.7063751, 161.1622925
22: -102.8709869, 62.0754356, -103.2174988, 62.1832352, -165.0542297, 165.2929382
23: -82.0602875, 61.6243210, -82.4290085, 61.6948738, -143.7551575, 144.0533295
24: -101.4163513, 63.0801849, -101.7190933, 63.1270218, -164.5433655, 164.7992554
25: -87.1449738, 66.0443802, -87.3771362, 66.1176147, -153.2625732, 153.4215088
26: -116.7188110, 94.0597534, -117.4057693, 94.2222443, -210.9410553, 211.4655151
27: -101.4186630, 66.7852783, -101.7693176, 66.8294678, -168.2481384, 168.5545959
28: -80.3013153, 67.8873138, -80.6242371, 67.9449615, -148.2462769, 148.5115356
29: -107.1628723, 58.2703819, -107.6010590, 58.3566170, -165.5194855, 165.8714142
30: -101.0300217, 74.2322693, -101.4286499, 74.3356171, -175.3656311, 175.6609192
31: -107.1243591, 67.0961609, -107.5877304, 67.1647720, -174.2891235, 174.6838684
32: -105.5402679, 69.3717804, -105.6890869, 69.4765167, -175.0167694, 175.0608521
33: -139.5725708, 92.2341156, -139.7101746, 92.5246582, -232.0972290, 231.9442749
34: -118.7504730, 64.0609283, -118.8797836, 64.2124939, -182.9629669, 182.9407043
35: -115.7563171, 71.7046204, -115.8664169, 71.8550491, -187.6113586, 187.5710449
36: -113.6829681, 71.6725464, -113.8198013, 71.7720795, -185.4550476, 185.4923401
37: -167.4313965, 74.2281036, -167.7463379, 74.3335266, -241.7649231, 241.9744263
38: -133.8702240, 83.3171997, -134.0610504, 83.4632874, -217.3335114, 217.3782501
39: -157.5440979, 87.6501923, -157.6931000, 87.8791046, -245.4232025, 245.3432922
40: -124.2653885, 71.9781952, -124.4536591, 72.1089783, -196.3743591, 196.4318542
41: -112.1335373, 81.0040588, -112.2866364, 81.1087646, -193.2423096, 193.2906952
42: -79.2996216, 72.3307266, -79.4627304, 72.4690857, -151.7686920, 151.7934570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9885538, upper bound: 94.0256812
time: 228.22 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0256812
time: 249.99 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -132.0200806, 89.1720734, -131.8710938, 88.8898010, -220.9098511, 221.0431671
1: -68.1921005, 68.7457275, -68.1027069, 68.4980545, -136.6901550, 136.8484344
2: -61.4211884, 71.2581940, -61.3164597, 70.9293747, -132.3505554, 132.5746460
3: -70.1713562, 83.8771820, -70.0657959, 83.5001984, -153.6715393, 153.9429626
4: -76.5068436, 82.6030960, -76.3853378, 82.2790298, -158.7858734, 158.9884338
5: -68.1272888, 86.1883240, -68.0092926, 85.7796783, -153.9069519, 154.1976166
6: -108.3170929, 80.7802353, -108.1719818, 80.6789551, -188.9960480, 188.9522095
7: -80.1852188, 81.2919769, -80.0345306, 80.9596024, -161.1448212, 161.3265076
8: -89.6247406, 102.5318604, -89.5166092, 102.1636047, -191.7883453, 192.0484619
9: -75.7570038, 80.5549927, -75.5407639, 80.4548721, -156.2118835, 156.0957642
10: -111.5605545, 104.8485718, -110.9775696, 104.7056656, -216.2662201, 215.8261108
11: -105.5760956, 64.6278687, -105.1599808, 64.5173492, -170.0934296, 169.7878418
12: -111.0873337, 85.8489227, -110.3898239, 85.6921463, -196.7794647, 196.2387390
13: -108.2995224, 109.7754517, -108.1882553, 109.5982437, -217.8977509, 217.9636993
14: -167.9454651, 95.2351608, -167.4585876, 95.1179276, -263.0633850, 262.6937561
15: -88.3814774, 76.4726410, -88.2357788, 76.2980804, -164.6795502, 164.7084198
16: -111.7505493, 80.2594147, -111.4741745, 80.1284790, -191.8790283, 191.7335815
17: -159.8361206, 83.9385452, -159.3991394, 83.7912292, -243.6273041, 243.3376770
18: -107.6587601, 79.3143539, -107.3833771, 79.2010269, -186.8597870, 186.6977234
19: -81.1603394, 48.6883507, -80.9148331, 48.6238441, -129.7841797, 129.6031799
20: -77.0615692, 60.7916374, -76.8534393, 60.7310143, -137.7925873, 137.6450806
21: -100.8020020, 60.3272438, -100.4528198, 60.2340508, -161.0360565, 160.7800598
22: -103.0758209, 62.2094383, -102.9219208, 62.0830650, -165.1588898, 165.1313477
23: -82.2693176, 61.6735001, -82.0963821, 61.5931625, -143.8624878, 143.7698669
24: -101.5733185, 63.0692596, -101.4665604, 63.0086365, -164.5819550, 164.5358276
25: -87.3097992, 66.1599426, -87.1783142, 66.0653763, -153.3751831, 153.3382568
26: -117.2427292, 94.3767624, -116.7720032, 94.2055435, -211.4482422, 211.1487732
27: -101.6343613, 66.8280182, -101.5097427, 66.7272339, -168.3616028, 168.3377686
28: -80.4408646, 67.9239426, -80.3384094, 67.8378143, -148.2786865, 148.2623596
29: -107.3930283, 58.4060020, -107.1996460, 58.3021240, -165.6951447, 165.6056519
30: -101.2215347, 74.3277130, -101.0631332, 74.2133636, -175.4349060, 175.3908386
31: -107.4611816, 67.1814117, -107.1815567, 67.1023560, -174.5635071, 174.3629761
32: -105.7938385, 69.5690460, -105.5702744, 69.4933777, -175.2872162, 175.1393127
33: -139.7743225, 92.5089035, -139.6581116, 92.2697144, -232.0440369, 232.1670227
34: -118.9356613, 64.2902908, -118.8470840, 64.0961456, -183.0317841, 183.1373749
35: -115.9784088, 71.9539108, -115.9061127, 71.7334442, -187.7118530, 187.8600159
36: -113.8384781, 71.7951202, -113.7469559, 71.6818085, -185.5202942, 185.5420837
37: -167.6662598, 74.3958511, -167.4895782, 74.2887878, -241.9550171, 241.8854370
38: -134.1017456, 83.5193176, -133.9628143, 83.3628998, -217.4646454, 217.4821167
39: -157.7876434, 87.8049011, -157.5922241, 87.6774063, -245.4650269, 245.3971252
40: -124.4815521, 72.1082382, -124.3246460, 72.0094147, -196.4909363, 196.4328918
41: -112.2947769, 81.1548615, -112.1787262, 81.0613403, -193.3561096, 193.3335876
42: -79.5529327, 72.5300598, -79.3366547, 72.4166718, -151.9696045, 151.8667145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9885538, upper bound: 94.0161149
time: 158.47 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0161149
time: 140.44 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -132.1104889, 89.1888733, -132.1284332, 89.2200928, -221.3305817, 221.3173065
1: -68.2602386, 68.7526398, -68.2750168, 68.7852325, -137.0454712, 137.0276489
2: -61.5325394, 71.2697296, -61.5295563, 71.3426208, -132.8751526, 132.7992859
3: -70.3012924, 83.8942490, -70.3091049, 83.9857330, -154.2870178, 154.2033539
4: -76.6231232, 82.6181183, -76.6135101, 82.6613159, -159.2844391, 159.2316284
5: -68.2436981, 86.2042923, -68.2368927, 86.3025894, -154.5462646, 154.4411926
6: -108.3494263, 80.7565765, -108.3691635, 80.7569885, -189.1064148, 189.1257324
7: -80.2840424, 81.2886124, -80.2851715, 81.3107681, -161.5948181, 161.5737762
8: -89.7209167, 102.5455322, -89.7200317, 102.5568390, -192.2777557, 192.2655640
9: -75.7957611, 80.5881348, -75.6752472, 80.6219177, -156.4176636, 156.2633820
10: -111.6005402, 104.9552765, -111.3302307, 105.0139694, -216.6145020, 216.2855072
11: -105.6036835, 64.7746735, -105.8231583, 64.7783661, -170.3820190, 170.5978394
12: -111.1048813, 86.0221481, -111.0477600, 86.0191269, -197.1240082, 197.0699158
13: -108.3305359, 109.8179626, -108.3287811, 110.0141068, -218.3446350, 218.1467285
14: -167.9909058, 95.3967285, -168.0177002, 95.3981476, -263.3890381, 263.4144287
15: -88.4305954, 76.5020981, -88.4234848, 76.5574265, -164.9880219, 164.9255829
16: -111.7981567, 80.2266693, -111.8047028, 80.2393875, -192.0375366, 192.0313721
17: -159.8684692, 84.0950394, -160.1477966, 84.0772247, -243.9456940, 244.2428284
18: -107.6882553, 79.4419250, -107.9326630, 79.4435883, -187.1318359, 187.3745880
19: -81.1891022, 48.7666740, -81.3665161, 48.7683983, -129.9575043, 130.1331940
20: -77.0920792, 60.8610306, -77.1581192, 60.8683853, -137.9604645, 138.0191498
21: -100.8345795, 60.4432907, -101.0237885, 60.4423599, -161.2769470, 161.4670715
22: -103.0859833, 62.2980766, -103.2755585, 62.2827835, -165.3687439, 165.5736237
23: -82.2928009, 61.7516479, -82.4859009, 61.7543793, -144.0471802, 144.2375488
24: -101.5927505, 63.1418152, -101.7821732, 63.1462326, -164.7389832, 164.9239807
25: -87.3233414, 66.2396469, -87.4228058, 66.2238312, -153.5471649, 153.6624451
26: -117.2597961, 94.5681915, -117.4733887, 94.5574722, -211.8172302, 212.0415802
27: -101.6739731, 66.8907623, -101.8777084, 66.8582687, -168.5322418, 168.7684631
28: -80.4726715, 67.9933167, -80.6902618, 67.9823456, -148.4550018, 148.6835785
29: -107.4051208, 58.5218315, -107.6493378, 58.5107918, -165.9159088, 166.1711426
30: -101.2453537, 74.4412766, -101.4837799, 74.4436035, -175.6889648, 175.9250488
31: -107.4981384, 67.2629242, -107.6768112, 67.2596893, -174.7578278, 174.9397278
32: -105.8256454, 69.5954285, -105.7498779, 69.6140900, -175.4397278, 175.3453064
33: -139.8641663, 92.5402603, -139.8817139, 92.5893250, -232.4534912, 232.4219666
34: -118.9645081, 64.3124008, -118.9937668, 64.2702942, -183.2347717, 183.3061523
35: -115.9844513, 71.9733963, -116.0073929, 71.9010162, -187.8854675, 187.9807739
36: -113.8488007, 71.8280029, -113.8870544, 71.8195648, -185.6683655, 185.7150421
37: -167.6950073, 74.4354858, -167.8273315, 74.4344864, -242.1294861, 242.2627869
38: -134.1344604, 83.5249100, -134.1749268, 83.5138245, -217.6482697, 217.6998291
39: -157.8267212, 87.8258591, -157.7799530, 87.9241028, -245.7508240, 245.6058044
40: -124.5207825, 72.0966949, -124.5464630, 72.1285706, -196.6493530, 196.6431580
41: -112.3290710, 81.1390457, -112.3615723, 81.1534805, -193.4825439, 193.5006104
42: -79.5808258, 72.5376740, -79.5246964, 72.5516205, -152.1324463, 152.0623474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=603, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=794, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 659
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9885537, upper bound: 94.0378213
time: 138.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0378213
time: 158.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 300.07 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 300.07
Output dim: 5, lower bound: -93.9885538, upper bound: 94.0039467
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 300.07
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0039467
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 300.07
Output dim: 5, lower bound: -93.9885538, upper bound: 94.0256812
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 300.07
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0256812
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 300.07
Output dim: 5, lower bound: -93.9885538, upper bound: 94.0161149
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 300.07
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0161149
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 300.07
Output dim: 5, lower bound: -93.9885537, upper bound: 94.0378213
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 300.07
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0378213

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -131.4913177, 88.7871552, -131.5369873, 88.8002930, -220.2916107, 220.3241119
1: -67.7991791, 68.4233627, -67.8377686, 68.4334030, -136.2325745, 136.2611389
2: -60.8753281, 70.8556824, -60.9282570, 70.8651733, -131.7405090, 131.7839355
3: -69.5216827, 83.3849182, -69.5910721, 83.4012527, -152.9229279, 152.9759827
4: -75.8911743, 82.1884460, -75.9533234, 82.2004776, -158.0916290, 158.1417694
5: -67.5153503, 85.6671371, -67.5757904, 85.6829834, -153.1983337, 153.2429199
6: -108.0022888, 80.5641861, -108.0257416, 80.5806885, -188.5829468, 188.5899200
7: -79.6008759, 80.8820190, -79.6538391, 80.8930130, -160.4938812, 160.5358582
8: -89.0842133, 102.0532150, -89.1400986, 102.0678101, -191.1520081, 191.1932983
9: -75.3924866, 80.1523132, -75.4057922, 80.1985550, -155.5910339, 155.5581055
10: -110.7834167, 103.8621597, -110.8086243, 103.9908829, -214.7742920, 214.6707764
11: -105.0070877, 63.9454765, -105.0263214, 64.0150299, -169.0221100, 168.9717712
12: -110.2567749, 84.8565979, -110.2744446, 84.9675903, -195.2243347, 195.1310425
13: -107.9476547, 109.3908768, -107.9729233, 109.4171600, -217.3648071, 217.3637848
14: -167.2051086, 94.4245758, -167.2378387, 94.5152893, -261.7203369, 261.6624146
15: -87.9069290, 76.1451874, -87.9512787, 76.1648483, -164.0717773, 164.0964661
16: -111.2512817, 79.8016052, -111.2794189, 79.8530121, -191.1042633, 191.0810242
17: -159.2370911, 83.2379227, -159.2553406, 83.3003540, -242.5374451, 242.4932556
18: -107.2089005, 78.7708130, -107.2320709, 78.8173828, -186.0262451, 186.0028687
19: -80.7688522, 48.3968353, -80.7882156, 48.4193802, -129.1882324, 129.1850586
20: -76.7010422, 60.4994621, -76.7210388, 60.5246811, -137.2257233, 137.2205048
21: -100.3013535, 59.8557358, -100.3207092, 59.8967819, -160.1981201, 160.1764526
22: -102.7774124, 61.8321609, -102.7968445, 61.8596992, -164.6371002, 164.6289978
23: -81.9674072, 61.3722496, -81.9842377, 61.3958321, -143.3632202, 143.3564758
24: -101.3217163, 62.8553009, -101.3427887, 62.8691483, -164.1908569, 164.1980896
25: -87.0632477, 65.7899551, -87.0785446, 65.8208008, -152.8840332, 152.8684998
26: -116.6060333, 93.4913177, -116.6280060, 93.5730667, -210.1791077, 210.1193237
27: -101.2799377, 66.6212616, -101.3122101, 66.6199493, -167.8998871, 167.9334564
28: -80.1899414, 67.7149582, -80.2086792, 67.7195892, -147.9095154, 147.9236450
29: -107.0771942, 57.9621887, -107.0927048, 57.9949188, -165.0720978, 165.0548859
30: -100.9326630, 73.8662338, -100.9489594, 73.9051666, -174.8378143, 174.8151855
31: -106.9830933, 66.8092499, -107.0096588, 66.8437195, -173.8267975, 173.8189087
32: -105.4092712, 69.2620850, -105.4307709, 69.2899475, -174.6992188, 174.6928406
33: -139.2637634, 92.1313629, -139.3129883, 92.1482697, -231.4120331, 231.4443512
34: -118.5855408, 63.9675217, -118.6255646, 63.9827919, -182.5683289, 182.5930786
35: -115.5758896, 71.6366272, -115.6275177, 71.6493149, -187.2252045, 187.2641449
36: -113.5250168, 71.5882416, -113.5627594, 71.5959702, -185.1209869, 185.1510010
37: -167.2918396, 74.0702057, -167.3197937, 74.0940781, -241.3859253, 241.3899841
38: -133.6705017, 83.2434006, -133.7161560, 83.2582245, -216.9287262, 216.9595642
39: -157.3637848, 87.5701599, -157.3938446, 87.5852509, -244.9490356, 244.9640045
40: -124.1110382, 71.9464111, -124.1400986, 71.9558258, -196.0668640, 196.0864868
41: -111.9989777, 80.9516754, -112.0240402, 80.9639130, -192.9628601, 192.9757080
42: -79.2038269, 72.2003174, -79.2209473, 72.2354965, -151.4393005, 151.4212646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=791, inp2_unstable=791, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9689953, upper bound: 94.0004244
time: 127.93 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9861724, upper bound: 94.0015550
time: 160.53 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -131.8054199, 89.0399094, -131.6285400, 88.8169403, -220.6223602, 220.6684570
1: -68.0064850, 68.7285538, -67.9112396, 68.4455872, -136.4520721, 136.6397858
2: -61.1217232, 71.2557220, -61.0415192, 70.8791809, -132.0009003, 132.2972412
3: -69.7974167, 83.9589691, -69.7318573, 83.4234161, -153.2208252, 153.6908264
4: -76.2003250, 82.5869064, -76.0864716, 82.2151566, -158.4154663, 158.6733704
5: -67.7796021, 86.2115326, -67.6945190, 85.7035828, -153.4831848, 153.9060364
6: -108.1950989, 80.6469193, -108.0582733, 80.5305405, -188.7256470, 188.7051697
7: -79.8882904, 81.2514191, -79.7545776, 80.8945847, -160.7828674, 161.0059967
8: -89.3207474, 102.4819336, -89.2393799, 102.0894623, -191.4102173, 191.7213135
9: -75.5546265, 80.3350906, -75.4239960, 80.2332306, -155.7878418, 155.7590942
10: -111.5689011, 104.3139725, -110.8544312, 104.2001419, -215.7690430, 215.1683960
11: -105.7876740, 64.2847061, -105.0488434, 64.1996155, -169.9872742, 169.3335266
12: -111.0042648, 85.2449112, -110.2978973, 85.1620483, -196.1663208, 195.5427856
13: -108.1271210, 109.9061661, -108.0034714, 109.4624405, -217.5895538, 217.9096375
14: -168.0242462, 94.7853470, -167.2894592, 94.7266235, -262.7508545, 262.0747986
15: -88.1362686, 76.4203033, -87.9885712, 76.1953278, -164.3315735, 164.4088745
16: -111.7208557, 79.9944611, -111.3287964, 79.8599167, -191.5807495, 191.3232574
17: -160.0612183, 83.5755386, -159.2869873, 83.4706116, -243.5318298, 242.8625183
18: -107.8797150, 79.0605774, -107.2729187, 78.9628067, -186.8425293, 186.3334656
19: -81.3034897, 48.5573158, -80.8223190, 48.4998093, -129.8032990, 129.3796082
20: -77.0587769, 60.6551437, -76.7565155, 60.5950775, -137.6538544, 137.4116516
21: -100.9671783, 60.1005783, -100.3543549, 60.0236359, -160.9908142, 160.4549255
22: -103.0912476, 62.0374413, -102.8102646, 61.9376564, -165.0289001, 164.8477020
23: -82.4173965, 61.5775337, -82.0110931, 61.4852142, -143.9026184, 143.5886230
24: -101.6969147, 62.9771042, -101.3624420, 62.9162941, -164.6132050, 164.3395386
25: -87.4098206, 65.9824982, -87.0940781, 65.9109192, -153.3207397, 153.0765686
26: -117.3417511, 93.9081268, -116.6500015, 93.7755432, -211.1172943, 210.5581360
27: -101.5371628, 66.6852264, -101.3518982, 66.6102142, -168.1473694, 168.0371246
28: -80.4435959, 67.8685760, -80.2448578, 67.7655792, -148.2091675, 148.1134338
29: -107.4894257, 58.1800346, -107.1043549, 58.0983200, -165.5877380, 165.2843933
30: -101.3859787, 74.1599274, -100.9708099, 74.0411758, -175.4271545, 175.1307220
31: -107.6532288, 67.0149689, -107.0581284, 66.9546967, -174.6079254, 174.0730896
32: -105.5696335, 69.3841400, -105.4518127, 69.2979736, -174.8675995, 174.8359528
33: -139.5498352, 92.5531006, -139.4249115, 92.1759949, -231.7258301, 231.9780121
34: -118.7698593, 64.2523956, -118.6829071, 64.0103455, -182.7801971, 182.9353027
35: -115.7633133, 71.9754791, -115.6910248, 71.6711273, -187.4344482, 187.6664886
36: -113.6985397, 71.8325272, -113.6175537, 71.6197891, -185.3183289, 185.4500732
37: -167.5887299, 74.2167740, -167.3559570, 74.1002655, -241.6889801, 241.5727234
38: -133.9333344, 83.5604858, -133.7769775, 83.2820358, -217.2153625, 217.3374329
39: -157.6176147, 87.8266449, -157.4383850, 87.6069183, -245.2245331, 245.2650299
40: -124.3641891, 72.1342316, -124.1890106, 71.9554825, -196.3196716, 196.3232422
41: -112.1646271, 81.1168289, -112.0549927, 80.9688797, -193.1334991, 193.1718140
42: -79.4039154, 72.3768768, -79.2478333, 72.2249069, -151.6288147, 151.6246948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=792, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9826156, upper bound: 94.0004244
time: 143.38 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9998094, upper bound: 94.0015550
time: 287.92 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -131.5803375, 88.8034286, -131.7943420, 89.1311340, -220.7114563, 220.5977478
1: -67.8662720, 68.4296951, -68.0096893, 68.7208099, -136.5870819, 136.4393921
2: -60.9855576, 70.8666077, -61.1407890, 71.2785492, -132.2640991, 132.0073853
3: -69.6505432, 83.4014969, -69.8338699, 83.8872681, -153.5377808, 153.2353668
4: -76.0053711, 82.2029266, -76.1815872, 82.5831528, -158.5885315, 158.3845215
5: -67.6306305, 85.6825104, -67.8028564, 86.2064056, -153.8370361, 153.4853516
6: -108.0329285, 80.5399780, -108.2225037, 80.6569519, -188.6898804, 188.7624817
7: -79.6983032, 80.8778381, -79.9048004, 81.2442627, -160.9425659, 160.7826385
8: -89.1791534, 102.0659180, -89.3430328, 102.4611969, -191.6403503, 191.4089355
9: -75.4300537, 80.1844101, -75.5392761, 80.3667908, -155.7968445, 155.7236786
10: -110.8223724, 103.9661331, -111.1615067, 104.2971573, -215.1195221, 215.1276093
11: -105.0338669, 64.0890808, -105.6899414, 64.2752151, -169.3090668, 169.7790222
12: -110.2737885, 85.0287323, -110.9332275, 85.2939987, -195.5677795, 195.9619446
13: -107.9742813, 109.4310837, -108.1094284, 109.8361130, -217.8103943, 217.5405121
14: -167.2498627, 94.5856323, -167.7982483, 94.7953491, -262.0452271, 262.3838501
15: -87.9441681, 76.1728821, -88.1274643, 76.4257050, -164.3698730, 164.3003540
16: -111.2971725, 79.7633514, -111.6090317, 79.9593658, -191.2565308, 191.3723450
17: -159.2686462, 83.3934937, -160.0044556, 83.5863647, -242.8549805, 243.3979492
18: -107.2368393, 78.8971863, -107.7825623, 79.0592346, -186.2960815, 186.6797485
19: -80.7969666, 48.4741135, -81.2401276, 48.5634575, -129.3604126, 129.7142334
20: -76.7313309, 60.5680923, -77.0261993, 60.6617470, -137.3930664, 137.5942993
21: -100.3333435, 59.9707603, -100.8921814, 60.1047363, -160.4380493, 160.8629456
22: -102.7865601, 61.9194984, -103.1515961, 62.0596581, -164.8462219, 165.0710907
23: -81.9902344, 61.4487114, -82.3742218, 61.5557556, -143.5459900, 143.8229370
24: -101.3402405, 62.9274483, -101.6597290, 63.0066910, -164.3469238, 164.5871582
25: -87.0763016, 65.8684082, -87.3233032, 65.9789276, -153.0552368, 153.1917114
26: -116.6219101, 93.6819000, -117.3298340, 93.9244232, -210.5462952, 211.0117188
27: -101.3167725, 66.6843567, -101.6905212, 66.7487793, -168.0655518, 168.3748779
28: -80.2207336, 67.7833405, -80.5610886, 67.8629608, -148.0836945, 148.3444214
29: -107.0888824, 58.0766678, -107.5432434, 58.2038116, -165.2926941, 165.6199036
30: -100.9549408, 73.9759598, -101.3700409, 74.1334686, -175.0883942, 175.3459930
31: -107.0191879, 66.8894577, -107.5058594, 67.0007858, -174.0199738, 174.3953247
32: -105.4404373, 69.2872772, -105.6106415, 69.4093781, -174.8498230, 174.8979187
33: -139.3523102, 92.1623840, -139.5363770, 92.4681168, -231.8203735, 231.6987610
34: -118.6131134, 63.9891357, -118.7714386, 64.1552277, -182.7683411, 182.7605743
35: -115.5786057, 71.6557083, -115.7261353, 71.8162689, -187.3948669, 187.3818359
36: -113.5346451, 71.6210556, -113.7029114, 71.7312775, -185.2658997, 185.3239746
37: -167.3189087, 74.1065979, -167.6580505, 74.2370605, -241.5559692, 241.7646179
38: -133.7011566, 83.2488861, -133.9275055, 83.4090576, -217.1102142, 217.1763763
39: -157.4023132, 87.5903244, -157.5811462, 87.8320999, -245.2344055, 245.1714783
40: -124.1486053, 71.9344635, -124.3614731, 72.0735626, -196.2221680, 196.2959290
41: -112.0321808, 80.9352722, -112.2068634, 81.0542221, -193.0863953, 193.1421356
42: -79.2313385, 72.2043991, -79.4088440, 72.3689270, -151.6002502, 151.6132355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 659
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9689953, upper bound: 94.0221777
time: 1293.06 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -93.9861724, upper bound: 94.0232839
time: 213.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 1508.90 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1508.90
Output dim: 5, lower bound: -93.9689953, upper bound: 94.0004244
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1508.90
Output dim: 5, lower bound: -93.9861724, upper bound: 94.0015550
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1508.90
Output dim: 5, lower bound: -93.9826156, upper bound: 94.0004244
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1508.90
Output dim: 5, lower bound: -93.9998094, upper bound: 94.0015550
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1508.90
Output dim: 5, lower bound: -93.9689953, upper bound: 94.0221777
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1508.90
Output dim: 5, lower bound: -93.9861724, upper bound: 94.0232839
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1508.90
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0256812
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1508.90
Output dim: 5, lower bound: -93.9885538, upper bound: 94.0161149
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1508.90
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0161149
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1508.90
Output dim: 5, lower bound: -93.9885537, upper bound: 94.0378213
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1508.90
Output dim: 5, lower bound: -94.0022002, upper bound: 94.0378213
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=154.09011840820312
rel_dist={5: [-94.06215389057739, 94.06215389943473]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 13195.21 seconds
