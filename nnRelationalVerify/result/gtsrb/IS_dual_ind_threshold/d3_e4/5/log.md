## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 5)
Time budget: 7200 seconds
Split limit: 100
Threshold: 98.0557179282


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.87 + 202.44 = 205.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -98.1538718, upper bound: 98.1538718

# Indivdual Split (IS) starts

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0759516, upper bound: 98.1409406
time: 165.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0759516, upper bound: 98.1484935
time: 145.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 311.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 311.42
Output dim: 5, lower bound: -98.0759516, upper bound: 98.1409406
IS_A2, status: Status.UNKNOWN, split count: 1, time: 311.42
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

Time for backsubstitution: 2.21 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0657578, upper bound: 98.0715968
time: 568.81 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0657578, upper bound: 98.1303956
time: 162.75 seconds

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

Time for backsubstitution: 2.20 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1656

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0657578, upper bound: 98.0785668
time: 141.52 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0657578, upper bound: 98.1371653
time: 174.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 318.33 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 318.33
Output dim: 5, lower bound: -98.0657578, upper bound: 98.0715968
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 318.33
Output dim: 5, lower bound: -98.0657578, upper bound: 98.1303956
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 318.33
Output dim: 5, lower bound: -98.0657578, upper bound: 98.0785668
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 318.33
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

Time for backsubstitution: 2.40 seconds

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
time: 153.13 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0581354
time: 190.86 seconds

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

Time for backsubstitution: 2.32 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
time: 149.00 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
time: 153.82 seconds

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

Time for backsubstitution: 2.33 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0676725
time: 138.82 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0677504
time: 167.20 seconds

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

Time for backsubstitution: 2.41 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1270118
time: 189.98 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1271070
time: 168.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 361.00 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 361.00
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0581354
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 361.00
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0581354
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 361.00
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 361.00
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1177430
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 361.00
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0676725
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 361.00
Output dim: 5, lower bound: -98.0111562, upper bound: 98.0677504
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 361.00
Output dim: 5, lower bound: -98.0111562, upper bound: 98.1270118
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 361.00
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

Time for backsubstitution: 2.39 seconds

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
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
time: 156.58 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
time: 189.13 seconds

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

Time for backsubstitution: 2.50 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
time: 325.63 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -98.0080233, upper bound: 98.0550070
time: 143.27 seconds

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

Time for backsubstitution: 2.33 seconds

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
time: 154.75 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
time: 229.23 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -131.9099731, 89.0621567, -131.9773712, 89.1745453, -221.0845184, 221.0395203
1: -68.0849915, 68.7411804, -68.1552429, 68.7532959, -136.8382874, 136.8964233
2: -61.2467690, 71.2723083, -61.3555679, 71.3121185, -132.5588837, 132.6278687
3: -69.9427643, 83.9817200, -70.0998383, 83.9388199, -153.8815918, 154.0815430
4: -76.3304138, 82.6073303, -76.4234161, 82.6232910, -158.9536896, 159.0307312
5: -67.9096222, 86.2326965, -68.0349503, 86.2555237, -154.1651459, 154.2676392
6: -108.2334747, 80.6509094, -108.2992554, 80.6728897, -188.9063721, 188.9501648
7: -79.9995728, 81.2582626, -80.1068954, 81.2724075, -161.2719727, 161.3651428
8: -89.4282532, 102.5032959, -89.5455322, 102.5114365, -191.9396667, 192.0488281
9: -75.6043320, 80.3792953, -75.6016388, 80.4841614, -156.0885010, 155.9809265
10: -111.6160126, 104.4429855, -111.2540512, 104.6961212, -216.3121185, 215.6970367
11: -105.8236771, 64.4494324, -105.7541580, 64.5776367, -170.4013062, 170.2035828
12: -111.0315323, 85.4389648, -110.9926758, 85.6841507, -196.7156830, 196.4316406
13: -108.1981049, 109.9551468, -108.2206268, 109.9302902, -218.1283875, 218.1757660
14: -168.0844879, 94.9640045, -167.9147949, 95.1496277, -263.2341309, 262.8787537
15: -88.2073898, 76.4561310, -88.2718506, 76.4946899, -164.7020874, 164.7279816
16: -111.7780380, 80.0074387, -111.7145767, 80.0816193, -191.8596497, 191.7220154
17: -160.1065826, 83.7496643, -160.0798340, 83.8711472, -243.9777222, 243.8294983
18: -107.9169617, 79.2045288, -107.8652267, 79.2937927, -187.2107544, 187.0697327
19: -81.3378601, 48.6458435, -81.3089600, 48.6909637, -130.0288239, 129.9548035
20: -77.0951157, 60.7340050, -77.0976181, 60.7829437, -137.8780518, 137.8316193
21: -101.0065613, 60.2306099, -100.9634857, 60.3093681, -161.3159027, 161.1940918
22: -103.1155243, 62.1364937, -103.2094574, 62.1999283, -165.3154602, 165.3459473
23: -82.4467316, 61.6662178, -82.4331818, 61.6917534, -144.1384735, 144.0993958
24: -101.7242050, 63.0591927, -101.7205048, 63.0849037, -164.8091125, 164.7796936
25: -87.4324951, 66.0717468, -87.3709564, 66.1252289, -153.5577087, 153.4427032
26: -117.3726883, 94.1224976, -117.3994904, 94.2811432, -211.6538086, 211.5219727
27: -101.5834045, 66.7671509, -101.7883072, 66.7814941, -168.3648987, 168.5554504
28: -80.4808807, 67.9483490, -80.6320190, 67.9365845, -148.4174500, 148.5803680
29: -107.5141754, 58.3080864, -107.5934296, 58.3818092, -165.8959808, 165.9015198
30: -101.4171906, 74.2897644, -101.4267731, 74.3421021, -175.7592926, 175.7165375
31: -107.6970215, 67.1072922, -107.6001587, 67.1687775, -174.8657837, 174.7074432
32: -105.6087646, 69.4258041, -105.6748352, 69.4881287, -175.0968933, 175.1006470
33: -139.6536102, 92.5907440, -139.7343903, 92.5315018, -232.1851196, 232.3251343
34: -118.8097687, 64.2826157, -118.8914337, 64.2166901, -183.0264587, 183.1740417
35: -115.7972107, 71.9982529, -115.8797379, 71.8615952, -187.6588135, 187.8779907
36: -113.7211914, 71.8689804, -113.8095856, 71.7796631, -185.5008545, 185.6785583
37: -167.6271973, 74.2764359, -167.7452698, 74.3192291, -241.9464264, 242.0216980
38: -133.9813385, 83.5785675, -134.0630341, 83.4638367, -217.4451599, 217.6416016
39: -157.6684418, 87.8531189, -157.6824341, 87.8840256, -245.5524597, 245.5355530
40: -124.4110413, 72.1376953, -124.4619064, 72.0995331, -196.5105743, 196.5995941
41: -112.2050400, 81.1256485, -112.2835083, 81.1004181, -193.3054504, 193.4091492
42: -79.4374237, 72.4078827, -79.4705353, 72.4496613, -151.8870850, 151.8784180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

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
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1657
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
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1667
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
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
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
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 854
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
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1369
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
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
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
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1430
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
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
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
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
time: 195.22 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0080233, upper bound: 98.1145772
time: 197.30 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -131.8905334, 89.1329346, -131.8000488, 88.8691254, -220.7596588, 220.9329834
1: -68.0845947, 68.7146606, -68.0453796, 68.4825668, -136.5671539, 136.7600403
2: -61.2839012, 71.2263947, -61.2383080, 70.9133911, -132.1972961, 132.4647064
3: -69.9803238, 83.8282776, -69.9635468, 83.4768066, -153.4571228, 153.7918243
4: -76.3385773, 82.5669861, -76.2933197, 82.2611694, -158.5997467, 158.8602905
5: -67.9722443, 86.1420746, -67.9222870, 85.7572784, -153.7295227, 154.0643311
6: -108.2504959, 80.7377472, -108.1397629, 80.6635895, -188.9140625, 188.8775024
7: -80.0490112, 81.2579346, -79.9581451, 80.9442978, -160.9932861, 161.2160797
8: -89.4935150, 102.4834595, -89.4453506, 102.1393433, -191.6328583, 191.9288025
9: -75.6987076, 80.4755859, -75.5045776, 80.4186707, -156.1173706, 155.9801636
10: -111.4871902, 104.5214996, -110.9395905, 104.5517426, -216.0389099, 215.4610901
11: -105.5176392, 64.3911896, -105.1293030, 64.3892365, -169.9068756, 169.5204926
12: -111.0304794, 85.5947113, -110.3618851, 85.5566559, -196.5871277, 195.9565887
13: -108.1462631, 109.6997299, -108.1088181, 109.5590363, -217.7052612, 217.8085327
14: -167.8402710, 94.9545746, -167.4048767, 94.9702225, -262.8104248, 262.3594360
15: -88.2315826, 76.4142151, -88.1677933, 76.2685699, -164.5001221, 164.5820007
16: -111.6589508, 80.1225128, -111.4269409, 80.0713654, -191.7303162, 191.5494537
17: -159.7664185, 83.7201996, -159.3615875, 83.6661072, -243.4325104, 243.0817871
18: -107.5847321, 79.1216965, -107.3460312, 79.0921173, -186.6768494, 186.4677277
19: -81.1036682, 48.5859451, -80.8857727, 48.5641136, -129.6677856, 129.4717102
20: -77.0039215, 60.6951828, -76.8231812, 60.6761284, -137.6800385, 137.5183716
21: -100.7417068, 60.1690636, -100.4215546, 60.1431160, -160.8847961, 160.5906067
22: -103.0155563, 62.1070480, -102.8927917, 62.0288315, -165.0443878, 164.9998474
23: -82.2188110, 61.5453300, -82.0704498, 61.5263596, -143.7451782, 143.6157837
24: -101.5174866, 62.9641800, -101.4418640, 62.9523582, -164.4698334, 164.4060364
25: -87.2602997, 66.0351562, -87.1537857, 65.9955444, -153.2558441, 153.1889343
26: -117.1729889, 94.1213226, -116.7375946, 94.0578156, -211.2308044, 210.8589172
27: -101.5589371, 66.7795258, -101.4733353, 66.6873779, -168.2463074, 168.2528687
28: -80.3832245, 67.8661194, -80.3080902, 67.7978973, -148.1811218, 148.1741943
29: -107.3424988, 58.2840462, -107.1744766, 58.2273521, -165.5698242, 165.4585114
30: -101.1679306, 74.1488342, -101.0355301, 74.1195526, -175.2874756, 175.1843567
31: -107.3848953, 67.0266571, -107.1421509, 67.0208664, -174.4057312, 174.1688080
32: -105.7202454, 69.5182571, -105.5326462, 69.4630585, -175.1832886, 175.0509033
33: -139.6171570, 92.4604645, -139.5734558, 92.2435455, -231.8607025, 232.0339203
34: -118.8248291, 64.2419434, -118.7954025, 64.0696869, -182.8945160, 183.0373535
35: -115.8343658, 71.9186707, -115.8400421, 71.7151489, -187.5495148, 187.7586975
36: -113.7058945, 71.7680893, -113.6903534, 71.6638184, -185.3697205, 185.4584351
37: -167.5811005, 74.3166351, -167.4490356, 74.2448730, -241.8259735, 241.7656708
38: -133.9649963, 83.4712982, -133.9018097, 83.3375549, -217.3025208, 217.3731079
39: -157.6809692, 87.7617874, -157.5380249, 87.6580582, -245.3390198, 245.2998047
40: -124.3943024, 72.0768204, -124.2809677, 71.9974823, -196.3917847, 196.3577881
41: -112.2169876, 81.1133041, -112.1406250, 81.0402603, -193.2572479, 193.2539062
42: -79.5064774, 72.4374084, -79.3121796, 72.3823090, -151.8887787, 151.7495728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

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
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
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
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1399
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
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1634
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
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
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
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1690
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
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1450
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
type: B, layer: 1, pos: 757
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
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 741
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
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 715
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
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1713
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
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1617
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
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0644636
time: 261.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0644636
time: 139.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -132.2158051, 89.3922577, -131.8410492, 88.8748627, -221.0906677, 221.2333069
1: -68.2994080, 69.0231323, -68.0784988, 68.4862518, -136.7856598, 137.1016235
2: -61.5386543, 71.6298828, -61.2910538, 70.9180603, -132.4567108, 132.9209290
3: -70.2647629, 84.4078979, -70.0287781, 83.4844818, -153.7492371, 154.4366760
4: -76.6557083, 82.9709320, -76.3564758, 82.2658386, -158.9215393, 159.3274078
5: -68.2439728, 86.6907196, -67.9773407, 85.7644958, -154.0084686, 154.6680603
6: -108.4527435, 80.8275986, -108.1523590, 80.6206512, -189.0733948, 188.9799500
7: -80.3440857, 81.6325226, -80.0045319, 80.9402924, -161.2843628, 161.6370544
8: -89.7381897, 102.9174500, -89.4917450, 102.1469421, -191.8851318, 192.4091949
9: -75.8622665, 80.6663971, -75.5060043, 80.4307098, -156.2929688, 156.1723938
10: -112.2775345, 104.9857559, -110.9590683, 104.6485825, -216.9261169, 215.9448242
11: -106.3104858, 64.7393494, -105.1365509, 64.4759674, -170.7864532, 169.8759003
12: -111.7840805, 85.9954376, -110.3689651, 85.6482391, -197.4323120, 196.3644104
13: -108.3392105, 110.2304306, -108.1031570, 109.5789032, -217.9180908, 218.3335876
14: -168.6718292, 95.3235168, -167.4235992, 95.0707092, -263.7425537, 262.7471008
15: -88.5127106, 76.6978912, -88.1876755, 76.2807617, -164.7934723, 164.8855591
16: -112.1387939, 80.3415146, -111.4467850, 80.0618744, -192.2006226, 191.7882996
17: -160.5987244, 84.0663834, -159.3715668, 83.7461548, -244.3448792, 243.4379578
18: -108.2645111, 79.4178696, -107.3625183, 79.1600037, -187.4244995, 186.7803955
19: -81.6426010, 48.7493210, -80.9000778, 48.6012421, -130.2438354, 129.6493988
20: -77.3662415, 60.8576241, -76.8385620, 60.7084236, -138.0746613, 137.6961670
21: -101.4131317, 60.4190254, -100.4349594, 60.2030296, -161.6161652, 160.8539734
22: -103.3371124, 62.3205795, -102.8929596, 62.0651817, -165.4022827, 165.2135315
23: -82.6761932, 61.7599525, -82.0813293, 61.5689087, -144.2450867, 143.8412781
24: -101.9013367, 63.0956688, -101.4465485, 62.9635086, -164.8648376, 164.5422211
25: -87.6132584, 66.2336731, -87.1567230, 66.0373993, -153.6506653, 153.3903809
26: -117.9178391, 94.5481262, -116.7422409, 94.1532898, -212.0711365, 211.2903748
27: -101.8238983, 66.8384399, -101.4900436, 66.6667404, -168.4906311, 168.3284912
28: -80.6424179, 68.0253830, -80.3238678, 67.8166046, -148.4590149, 148.3492432
29: -107.7641068, 58.5095215, -107.1750183, 58.2757912, -166.0398865, 165.6845398
30: -101.6300278, 74.4494934, -101.0427856, 74.1826477, -175.8126526, 175.4922791
31: -108.0673447, 67.2378311, -107.1636963, 67.0729599, -175.1402893, 174.4015198
32: -105.8813400, 69.6495743, -105.5354004, 69.4589386, -175.3402710, 175.1849670
33: -139.9111328, 92.8894577, -139.6253052, 92.2547913, -232.1659241, 232.5147552
34: -119.0167236, 64.5329437, -118.8185730, 64.0802231, -183.0969543, 183.3515167
35: -116.0321274, 72.2629089, -115.8608170, 71.7245407, -187.7566681, 188.1237183
36: -113.8913879, 72.0163956, -113.7133026, 71.6739960, -185.5653839, 185.7297058
37: -167.8940125, 74.4758606, -167.4613953, 74.2378693, -242.1318665, 241.9372253
38: -134.2362366, 83.7922058, -133.9233704, 83.3460083, -217.5822296, 217.7155762
39: -157.9441528, 88.0241241, -157.5520935, 87.6665344, -245.6106415, 245.5762177
40: -124.6572647, 72.2723846, -124.3019867, 71.9930420, -196.6502991, 196.5743713
41: -112.3909378, 81.2839661, -112.1505585, 81.0351257, -193.4260559, 193.4345245
42: -79.7216949, 72.6361618, -79.3236084, 72.3669205, -152.0886230, 151.9597778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=792, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

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
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1598
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
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 555
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
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 878
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
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
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
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 899
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
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1626
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
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 741
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
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 885
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
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1617
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0645564
time: 173.29 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0080233, upper bound: 98.0645564
time: 204.13 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -131.9292908, 89.1385880, -132.0570831, 89.1995621, -221.1288452, 221.1956787
1: -68.1143036, 68.7149124, -68.2172852, 68.7696838, -136.8839874, 136.9321899
2: -61.3346443, 71.2296448, -61.4509277, 71.3266296, -132.6612549, 132.6805725
3: -70.0401001, 83.8338776, -70.2063599, 83.9623413, -154.0024414, 154.0402222
4: -76.3913574, 82.5718689, -76.5213318, 82.6435242, -159.0348816, 159.0932007
5: -68.0257111, 86.1472397, -68.1494293, 86.2802048, -154.3059082, 154.2966614
6: -108.2640839, 80.7106171, -108.3370056, 80.7407761, -189.0048370, 189.0476227
7: -80.0937500, 81.2503662, -80.2085114, 81.2953644, -161.3891144, 161.4588776
8: -89.5376129, 102.4861679, -89.6483459, 102.5325012, -192.0701141, 192.1344910
9: -75.7115326, 80.4882584, -75.6385498, 80.5854874, -156.2970276, 156.1268005
10: -111.5037231, 104.5636520, -111.2921677, 104.8589401, -216.3626709, 215.8558197
11: -105.5271301, 64.4574509, -105.7925568, 64.6495285, -170.1766357, 170.2500000
12: -111.0343475, 85.6742020, -111.0198975, 85.8830719, -196.9174194, 196.6940918
13: -108.1377106, 109.7176895, -108.2472076, 109.9754105, -218.1130981, 217.9648590
14: -167.8558044, 95.0308762, -167.9640503, 95.2504120, -263.1062012, 262.9949341
15: -88.2341537, 76.4260254, -88.3506165, 76.5284271, -164.7625732, 164.7766418
16: -111.6782532, 80.0790405, -111.7573547, 80.1799316, -191.8581543, 191.8363800
17: -159.7760925, 83.7931976, -160.1101685, 83.9519043, -243.7279968, 243.9033661
18: -107.5950546, 79.1796494, -107.8955536, 79.3341980, -186.9292603, 187.0751953
19: -81.1154785, 48.6218948, -81.3373871, 48.7082825, -129.8237610, 129.9592743
20: -77.0167618, 60.7262230, -77.1278839, 60.8132172, -137.8299866, 137.8541107
21: -100.7551804, 60.2226448, -100.9925613, 60.3510323, -161.1062164, 161.2152100
22: -103.0128326, 62.1478462, -103.2465210, 62.2285652, -165.2413940, 165.3943634
23: -82.2277527, 61.5802269, -82.4600449, 61.6869659, -143.9147034, 144.0402679
24: -101.5224762, 62.9976425, -101.7573547, 63.0898628, -164.6123352, 164.7549896
25: -87.2622223, 66.0715485, -87.3982010, 66.1537018, -153.4159241, 153.4697571
26: -117.1740646, 94.2098694, -117.4389572, 94.4094086, -211.5834503, 211.6488342
27: -101.5752792, 66.8015900, -101.8422470, 66.8174438, -168.3927307, 168.6438293
28: -80.3967514, 67.8972778, -80.6600571, 67.9421997, -148.3389587, 148.5573425
29: -107.3419037, 58.3382492, -107.6242523, 58.4357300, -165.7776184, 165.9624939
30: -101.1759949, 74.1980896, -101.4562912, 74.3488846, -175.5248718, 175.6543884
31: -107.4000702, 67.0635223, -107.6373291, 67.1779175, -174.5779877, 174.7008514
32: -105.7331238, 69.5230484, -105.7123871, 69.5832062, -175.3163147, 175.2354431
33: -139.6560211, 92.4733276, -139.7967072, 92.5631485, -232.2191772, 232.2700348
34: -118.8344269, 64.2485886, -118.9418564, 64.2432175, -183.0776367, 183.1904449
35: -115.8233795, 71.9270325, -115.9395752, 71.8825531, -187.7059174, 187.8666077
36: -113.7051086, 71.7821503, -113.8303070, 71.8009033, -185.5059814, 185.6124573
37: -167.5899658, 74.3244705, -167.7867737, 74.3892212, -241.9791870, 242.1112213
38: -133.9743347, 83.4678116, -134.1132812, 83.4882431, -217.4625549, 217.5810852
39: -157.6947174, 87.7696152, -157.7254333, 87.9044495, -245.5991669, 245.4950409
40: -124.4100723, 72.0635376, -124.5026550, 72.1158524, -196.5258789, 196.5661926
41: -112.2318344, 81.0924988, -112.3235397, 81.1314545, -193.3632812, 193.4160461
42: -79.5184479, 72.4254608, -79.5003281, 72.5169678, -152.0354156, 151.9257812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=794, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

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
type: B, layer: 1, pos: 1655
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
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1657
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
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 556
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
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1667
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
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 632
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
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 1399
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
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 763
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
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 648
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
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 673
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
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
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
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 867
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
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 700
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
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1367
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
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 725
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
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 956
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
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 692
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
type: B, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1237671
time: 201.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1237671
time: 168.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -132.2553406, 89.3984070, -132.0989075, 89.2055206, -221.4608612, 221.4973145
1: -68.3302002, 69.0240021, -68.2516403, 68.7738342, -137.1040344, 137.2756348
2: -61.5906105, 71.6336670, -61.5051498, 71.3316956, -132.9223022, 133.1388245
3: -70.3255463, 84.4139404, -70.2726898, 83.9704437, -154.2959900, 154.6866302
4: -76.7101440, 82.9762726, -76.5850220, 82.6485901, -159.3587341, 159.5612793
5: -68.2984619, 86.6961670, -68.2056427, 86.2875214, -154.5859833, 154.9018097
6: -108.4668884, 80.8024902, -108.3500137, 80.7004395, -189.1673279, 189.1524963
7: -80.3898315, 81.6256409, -80.2559891, 81.2919235, -161.6817627, 161.8816223
8: -89.7834320, 102.9209671, -89.6961517, 102.5407410, -192.3241730, 192.6171265
9: -75.8752823, 80.6824646, -75.6414337, 80.5990753, -156.4743652, 156.3238983
10: -112.2947845, 105.0305862, -111.3124390, 104.9590988, -217.2538452, 216.3430176
11: -106.3208847, 64.8072205, -105.8002930, 64.7380371, -171.0589142, 170.6074982
12: -111.7884674, 86.0760040, -111.0273361, 85.9755402, -197.7640076, 197.1033173
13: -108.3431244, 110.2484970, -108.2495804, 109.9951935, -218.3383026, 218.4980774
14: -168.6883240, 95.3999329, -167.9836426, 95.3507690, -264.0390930, 263.3835449
15: -88.5290527, 76.7102966, -88.3788681, 76.5409698, -165.0700073, 165.0891418
16: -112.1587982, 80.3086853, -111.7777328, 80.1769409, -192.3357239, 192.0864258
17: -160.6093292, 84.1396637, -160.1209106, 84.0321350, -244.6414490, 244.2605438
18: -108.2758331, 79.4765396, -107.9127808, 79.4026794, -187.6785126, 187.3893127
19: -81.6550827, 48.7866364, -81.3522644, 48.7469025, -130.4019775, 130.1389008
20: -77.3795624, 60.8891830, -77.1435776, 60.8460846, -138.2256470, 138.0327606
21: -101.4273453, 60.4736633, -101.0065460, 60.4120293, -161.8393707, 161.4802094
22: -103.3355026, 62.3618774, -103.2476349, 62.2654076, -165.6009064, 165.6095123
23: -82.6855774, 61.7961960, -82.4712372, 61.7308121, -144.4163818, 144.2674255
24: -101.9070740, 63.1293373, -101.7628098, 63.1009560, -165.0080261, 164.8921509
25: -87.6159592, 66.2707291, -87.4017181, 66.1961899, -153.8121338, 153.6724548
26: -117.9200974, 94.6372070, -117.4443741, 94.5054550, -212.4255371, 212.0815430
27: -101.8412476, 66.8671265, -101.8594284, 66.8005753, -168.6418152, 168.7265625
28: -80.6562958, 68.0579987, -80.6760101, 67.9615631, -148.6178589, 148.7340088
29: -107.7643280, 58.5642433, -107.6252365, 58.4846878, -166.2489929, 166.1894531
30: -101.6388474, 74.5007172, -101.4639893, 74.4140167, -176.0528564, 175.9647064
31: -108.0833130, 67.2757263, -107.6595764, 67.2311401, -175.3144531, 174.9353027
32: -105.8947296, 69.6555862, -105.7153549, 69.5808182, -175.4755554, 175.3709412
33: -139.9506836, 92.9028320, -139.8487549, 92.5748825, -232.5255737, 232.7515869
34: -119.0288696, 64.5397110, -118.9675369, 64.2551270, -183.2839966, 183.5072479
35: -116.0312347, 72.2715073, -115.9738388, 71.8923645, -187.9235992, 188.2453308
36: -113.8921432, 72.0288239, -113.8542404, 71.8113403, -185.7034607, 185.8830566
37: -167.9037476, 74.4886780, -167.7996216, 74.3868103, -242.2905579, 242.2882996
38: -134.2493134, 83.7894745, -134.1382904, 83.4973297, -217.7466125, 217.9277649
39: -157.9586182, 88.0324554, -157.7405396, 87.9135284, -245.8721466, 245.7729950
40: -124.6737061, 72.2613831, -124.5239029, 72.1124649, -196.7861633, 196.7852783
41: -112.4063721, 81.2649155, -112.3337097, 81.1293335, -193.5357056, 193.5986023
42: -79.7338867, 72.6296387, -79.5119095, 72.5065460, -152.2404327, 152.1415405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=603, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

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
type: B, layer: 1, pos: 1671
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
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1657
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
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1667
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
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1673
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
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1654
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
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1368
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
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1590
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
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 648
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
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 673
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
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1722
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
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 899
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
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 530
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
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 700
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
type: B, layer: 1, pos: 757
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
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 725
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
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1430
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
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 512
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
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1617
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
type: B, layer: 1, pos: 1718

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1238848
time: 135.12 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.1238844, upper bound: 98.1238848
time: 251.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 389.00 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0550070
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 389.00
Output dim: 5, lower bound: -98.0080233, upper bound: 98.0550070
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1145772
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -98.0080233, upper bound: 98.1145772
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0644636
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0644636
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0645564
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -98.0080233, upper bound: 98.0645564
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1237671
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1237671
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1238848
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 389.00
Output dim: 5, lower bound: -98.1238844, upper bound: 98.1238848

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -131.5353394, 88.6950226, -131.6606140, 88.9607544, -220.4960938, 220.3556366
1: -67.8533630, 68.3513794, -67.9317932, 68.5904999, -136.4438629, 136.2831726
2: -60.9752312, 70.7830582, -61.1464386, 71.1399384, -132.1151733, 131.9295044
3: -69.6424866, 83.2294693, -69.7665710, 83.6109467, -153.2534180, 152.9960327
4: -75.9998474, 82.1374741, -76.2464371, 82.4716034, -158.4714355, 158.3839111
5: -67.6223297, 85.5273590, -67.7421112, 85.9553375, -153.5776672, 153.2694702
6: -107.9571457, 80.5430145, -108.1232834, 80.5841217, -188.5412598, 188.6662903
7: -79.6828232, 80.7241745, -79.8023605, 80.9901276, -160.6729431, 160.5265198
8: -89.1725159, 101.9150925, -89.2876434, 102.2042999, -191.3768158, 191.2027283
9: -75.4014206, 80.1654892, -75.4738998, 80.3912354, -155.7926636, 155.6393890
10: -110.7455444, 103.9536514, -111.0175629, 104.4303436, -215.1758728, 214.9712219
11: -104.9957657, 64.0710449, -105.5269928, 64.3970718, -169.3928223, 169.5980225
12: -110.0624008, 85.0138321, -110.5787048, 85.2816010, -195.3439941, 195.5925293
13: -107.9571075, 109.4079437, -108.0706558, 109.8153839, -217.7724915, 217.4785919
14: -167.2102051, 94.4277191, -167.5976410, 94.7399597, -261.9501648, 262.0253601
15: -87.9416046, 76.1263580, -88.1538010, 76.3576660, -164.2992706, 164.2801514
16: -111.2381134, 79.7739105, -111.4628525, 80.0090942, -191.2472076, 191.2367401
17: -159.2388611, 83.3464661, -159.7444458, 83.6461411, -242.8850098, 243.0908813
18: -107.2061462, 78.8814011, -107.7315140, 79.0640564, -186.2701874, 186.6129150
19: -80.7552643, 48.4693642, -81.1647186, 48.5917892, -129.3470459, 129.6340790
20: -76.7046890, 60.4975052, -76.9531784, 60.5990868, -137.3037720, 137.4506531
21: -100.2986603, 59.9515381, -100.7921829, 60.1646843, -160.4633484, 160.7437134
22: -102.7604370, 61.8980560, -103.0750427, 62.0625305, -164.8229523, 164.9730988
23: -81.9590302, 61.4134102, -82.2750549, 61.5466690, -143.5056915, 143.6884613
24: -101.3115692, 62.9128418, -101.6146622, 63.0048599, -164.3164062, 164.5274963
25: -87.0512924, 65.8307037, -87.2503204, 65.9699707, -153.0212708, 153.0810242
26: -116.5596085, 93.6579285, -117.2249374, 94.0276566, -210.5872345, 210.8828735
27: -101.2958603, 66.6162186, -101.6270447, 66.6301880, -167.9260559, 168.2432556
28: -80.1966095, 67.6980133, -80.4710388, 67.7336578, -147.9302673, 148.1690369
29: -107.0577164, 58.0681076, -107.3976135, 58.2649956, -165.3227081, 165.4657135
30: -100.9294052, 73.8768005, -101.2231445, 74.0555878, -174.9849854, 175.0999451
31: -106.9644928, 66.8737030, -107.4281693, 67.0186462, -173.9831238, 174.3018494
32: -105.3285522, 69.2778473, -105.4393463, 69.3434677, -174.6720123, 174.7171936
33: -139.2167969, 92.1419754, -139.4056549, 92.2833633, -231.5001221, 231.5476227
34: -118.5851669, 63.9728394, -118.7686310, 64.0945282, -182.6796875, 182.7414551
35: -115.5424728, 71.6426392, -115.7258148, 71.7443085, -187.2867737, 187.3684387
36: -113.4874573, 71.6067963, -113.6658478, 71.6686478, -185.1560974, 185.2726440
37: -167.0971680, 74.1052399, -167.3069305, 74.0120468, -241.1092224, 241.4121704
38: -133.6638489, 83.2278976, -133.9217072, 83.2990875, -216.9629364, 217.1495972
39: -157.2114563, 87.5725021, -157.2935638, 87.5653229, -244.7767792, 244.8660583
40: -124.0153809, 71.9274139, -124.1695938, 71.8298340, -195.8452148, 196.0970154
41: -111.8830566, 80.9355621, -111.9836884, 80.9017944, -192.7848511, 192.9192352
42: -79.1538086, 72.2020035, -79.2929535, 72.3449097, -151.4987183, 151.4949646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=602, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=791, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1718
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
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
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
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1565
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
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 933
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
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 916
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
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1743
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
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1564
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
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1558
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
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1585
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
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 565
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
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 971
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
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1710
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
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 531
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
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 725
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
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1448
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
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1430
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
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 810
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
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 943
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
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.8949099, upper bound: 98.1057829
time: 135.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.8949099, upper bound: 98.1075801
time: 198.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -131.5877380, 88.8004303, -131.9267731, 89.1560059, -220.7437439, 220.7272034
1: -67.8735657, 68.4280396, -68.1183853, 68.7377853, -136.6113434, 136.5464172
2: -60.9959221, 70.8637314, -61.2984695, 71.2950745, -132.2910004, 132.1622009
3: -69.6624374, 83.3936996, -70.0304108, 83.9093170, -153.5717468, 153.4241028
4: -76.0165329, 82.1985626, -76.3578873, 82.6031189, -158.6196442, 158.5564575
5: -67.6413422, 85.6763306, -67.9760895, 86.2299957, -153.8713379, 153.6524048
6: -108.0267181, 80.5636978, -108.2680511, 80.7098541, -188.7365570, 188.8317261
7: -79.7075653, 80.8751755, -80.0561218, 81.2557983, -160.9633484, 160.9312897
8: -89.1867599, 102.0610657, -89.4942017, 102.4838409, -191.6705780, 191.5552673
9: -75.4358063, 80.1884003, -75.5903854, 80.4665680, -155.9023743, 155.7787781
10: -110.8087158, 103.9837189, -111.1996155, 104.5927048, -215.4013977, 215.1833344
11: -105.0350037, 64.0993423, -105.7378464, 64.4774017, -169.5124054, 169.8371887
12: -110.2678070, 85.0461197, -110.9601135, 85.5895309, -195.8573303, 196.0062256
13: -107.9999466, 109.4339142, -108.2113037, 109.9056015, -217.9055481, 217.6452026
14: -167.2566223, 94.5905762, -167.8848724, 95.0300903, -262.2867126, 262.4754639
15: -87.9694290, 76.1664276, -88.2643890, 76.4623413, -164.4317627, 164.4308014
16: -111.2977066, 79.7982483, -111.6813431, 80.0869598, -191.3846741, 191.4795837
17: -159.2757721, 83.3918304, -160.0612488, 83.7597809, -243.0355530, 243.4530792
18: -107.2392349, 78.9087830, -107.8397369, 79.2190857, -186.4583130, 186.7485199
19: -80.7922668, 48.4810448, -81.2780914, 48.6484833, -129.4407349, 129.7591400
20: -76.7329025, 60.5716476, -77.0764923, 60.7418671, -137.4747620, 137.6481323
21: -100.3338165, 59.9777718, -100.9405975, 60.2382889, -160.5721130, 160.9183655
22: -102.7952499, 61.9265938, -103.2017670, 62.1582909, -164.9535370, 165.1283569
23: -81.9916840, 61.4542656, -82.4162140, 61.6412735, -143.6329651, 143.8704834
24: -101.3442078, 62.9336624, -101.7104721, 63.0734177, -164.4176178, 164.6441345
25: -87.0804443, 65.8696365, -87.3615417, 66.0701141, -153.1505585, 153.2311707
26: -116.6256180, 93.6997986, -117.3803940, 94.1790848, -210.8046722, 211.0801697
27: -101.3200989, 66.6883850, -101.7674103, 66.7865906, -168.1066895, 168.4557953
28: -80.2227936, 67.7850037, -80.6113052, 67.9055176, -148.1283112, 148.3963013
29: -107.0955811, 58.0873642, -107.5856400, 58.3306084, -165.4261780, 165.6730042
30: -100.9575348, 73.9841919, -101.4134140, 74.2639923, -175.2215271, 175.3975983
31: -107.0138779, 66.8962173, -107.5590973, 67.1100540, -174.1239319, 174.4553070
32: -105.4374847, 69.3000488, -105.6570587, 69.4882812, -174.9257507, 174.9570923
33: -139.3554993, 92.1653976, -139.6656799, 92.5162659, -231.8717651, 231.8310852
34: -118.6168594, 63.9948273, -118.8586884, 64.2002563, -182.8170776, 182.8535156
35: -115.5891876, 71.6566010, -115.8337097, 71.8486633, -187.4378357, 187.4903107
36: -113.5348587, 71.6247330, -113.7709274, 71.7658005, -185.3006592, 185.3956604
37: -167.3092499, 74.1213074, -167.7030029, 74.3212433, -241.6304932, 241.8243103
38: -133.7059021, 83.2570877, -134.0271454, 83.4493179, -217.1552124, 217.2842407
39: -157.3984375, 87.5933380, -157.6436310, 87.8720322, -245.2704315, 245.2369690
40: -124.1434708, 71.9446716, -124.4216614, 72.0999603, -196.2434082, 196.3663330
41: -112.0252075, 80.9559021, -112.2545242, 81.0985336, -193.1237183, 193.2104187
42: -79.2283401, 72.2232666, -79.4469757, 72.4645844, -151.6929169, 151.6702271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=602, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=794, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1718
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
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 750
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
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1565
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
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 876
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
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1719
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
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
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
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 737
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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1564
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
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1698
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
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 538
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
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1414
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
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1450
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
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1542
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
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 851
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
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 775
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
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 783
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
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1004
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
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 987
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
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 849

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.8949099, upper bound: 98.1057829
time: 171.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9363925, upper bound: 98.1075801
time: 165.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -131.8508148, 88.9485931, -131.6999817, 88.9654999, -220.8163147, 220.6485748
1: -68.0619354, 68.6575394, -67.9641190, 68.5940628, -136.6559906, 136.6216583
2: -61.2229233, 71.1841125, -61.1982613, 71.1442871, -132.3672028, 132.3823700
3: -69.9192886, 83.8041687, -69.8301926, 83.6179504, -153.5372314, 153.6343689
4: -76.3109283, 82.5367889, -76.3074341, 82.4758530, -158.7867737, 158.8442230
5: -67.8877640, 86.0723724, -67.7961426, 85.9616547, -153.8494263, 153.8685150
6: -108.1522522, 80.6273422, -108.1349716, 80.5423431, -188.6945953, 188.7622986
7: -79.9715805, 81.0948639, -79.8476791, 80.9857025, -160.9572754, 160.9425354
8: -89.4104614, 102.3453674, -89.3331985, 102.2115936, -191.6220398, 191.6785583
9: -75.5644226, 80.3527222, -75.4759674, 80.4028702, -155.9672852, 155.8286896
10: -111.5321274, 104.4084702, -111.0368958, 104.5263901, -216.0585175, 215.4453278
11: -105.7777252, 64.4120789, -105.5319977, 64.4822464, -170.2599487, 169.9440613
12: -110.8109055, 85.4032593, -110.5852737, 85.3703690, -196.1812744, 195.9885254
13: -108.1494675, 109.9250565, -108.0701904, 109.8333282, -217.9827881, 217.9952393
14: -168.0305176, 94.7886200, -167.6151276, 94.8384323, -262.8689575, 262.4037476
15: -88.1746674, 76.4035645, -88.1530151, 76.3690338, -164.5436859, 164.5565491
16: -111.7096176, 79.9792786, -111.4813461, 79.9972382, -191.7068481, 191.4606323
17: -160.0642395, 83.6844864, -159.7538757, 83.7240448, -243.7882843, 243.4383545
18: -107.8783875, 79.1720886, -107.7478790, 79.1302567, -187.0086365, 186.9199524
19: -81.2907104, 48.6313782, -81.1786575, 48.6296158, -129.9203186, 129.8100281
20: -77.0629120, 60.6538086, -76.9677124, 60.6300850, -137.6929932, 137.6215210
21: -100.9655075, 60.1974754, -100.8052063, 60.2242470, -161.1897278, 161.0026855
22: -103.0758972, 62.1041412, -103.0746307, 62.0977440, -165.1736450, 165.1787567
23: -82.4097977, 61.6205673, -82.2849579, 61.5892334, -143.9990234, 143.9055176
24: -101.6876297, 63.0352859, -101.6182709, 63.0111351, -164.6987610, 164.6535492
25: -87.3987579, 66.0243301, -87.2521439, 66.0109100, -153.4096680, 153.2764740
26: -117.2973785, 94.0755234, -117.2283630, 94.1209412, -211.4183197, 211.3038940
27: -101.5555878, 66.6874084, -101.6420898, 66.6125793, -168.1681519, 168.3294983
28: -80.4511185, 67.8537445, -80.4860229, 67.7519531, -148.2030640, 148.3397675
29: -107.4712296, 58.2866402, -107.3968658, 58.3125801, -165.7838135, 165.6835022
30: -101.3845825, 74.1730499, -101.2289124, 74.1180267, -175.5026093, 175.4019470
31: -107.6355515, 67.0806808, -107.4491348, 67.0705566, -174.7061157, 174.5298004
32: -105.4902191, 69.4010010, -105.4410782, 69.3389206, -174.8291321, 174.8420715
33: -139.5036926, 92.5643005, -139.4555359, 92.2936096, -231.7972717, 232.0198364
34: -118.7728043, 64.2573471, -118.7926559, 64.1054916, -182.8782654, 183.0500031
35: -115.7418976, 71.9820175, -115.7574768, 71.7534332, -187.4953156, 187.7395020
36: -113.6631165, 71.8487396, -113.6865387, 71.6787262, -185.3418427, 185.5352783
37: -167.3961639, 74.2580414, -167.3173828, 74.0062027, -241.4023743, 241.5754242
38: -133.9315796, 83.5457077, -133.9448853, 83.3074646, -217.2390289, 217.4906006
39: -157.4661713, 87.8297882, -157.3065186, 87.5733643, -245.0395355, 245.1363068
40: -124.2706451, 72.1175766, -124.1891022, 71.8247986, -196.0954285, 196.3066711
41: -112.0505753, 81.1022034, -111.9922333, 80.8985901, -192.9491272, 193.0944366
42: -79.3550491, 72.3839493, -79.3032608, 72.3255692, -151.6806183, 151.6871948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=602, inp2_unstable=602, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=792, inp2_unstable=792, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1672
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
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1718
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
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 984
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
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1565
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
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1555
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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1698
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
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1417
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
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1542
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
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 775
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
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 846
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
type: A, layer: 1, pos: 849

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 647

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9372108, upper bound: 97.9725590
time: 584.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9372108, upper bound: 98.0831958
time: 165.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 752.00 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 752.00
Output dim: 5, lower bound: -97.8949099, upper bound: 98.1057829
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 752.00
Output dim: 5, lower bound: -97.8949099, upper bound: 98.1075801
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 752.00
Output dim: 5, lower bound: -97.8949099, upper bound: 98.1057829
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 752.00
Output dim: 5, lower bound: -97.9363925, upper bound: 98.1075801
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 752.00
Output dim: 5, lower bound: -97.9372108, upper bound: 97.9725590
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 752.00
Output dim: 5, lower bound: -97.9372108, upper bound: 98.0831958
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -98.0080233, upper bound: 98.1145772
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0644636
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0644636
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.0645564
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -98.0080233, upper bound: 98.0645564
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1237671
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1237671
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -97.9664259, upper bound: 98.1238848
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 752.00
Output dim: 5, lower bound: -98.1238844, upper bound: 98.1238848

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 205.30 + 7258.93 = 7464.23 seconds
