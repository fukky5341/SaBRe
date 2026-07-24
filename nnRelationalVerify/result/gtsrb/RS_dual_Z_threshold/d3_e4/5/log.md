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
execution time: IAR + RelationalAnalysis = 2.88 + 201.07 = 203.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -98.1538718, upper bound: 98.1538718

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0879376, upper bound: 98.1484935
time: 411.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.1484935, upper bound: 98.0879376
time: 126.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 538.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 538.11
Output dim: 5, lower bound: -98.0879376, upper bound: 98.1484935
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 538.11
Output dim: 5, lower bound: -98.1484935, upper bound: 98.0879376

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0352239, upper bound: 98.1362214
time: 132.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0734214, upper bound: 98.0792169
time: 202.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0792169, upper bound: 98.0734214
time: 200.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.1362214, upper bound: 98.0352239
time: 368.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 571.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 571.04
Output dim: 5, lower bound: -98.0352239, upper bound: 98.1362214
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 571.04
Output dim: 5, lower bound: -98.0734214, upper bound: 98.0792169
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 571.04
Output dim: 5, lower bound: -98.0792169, upper bound: 98.0734214
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 571.04
Output dim: 5, lower bound: -98.1362214, upper bound: 98.0352239

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9660982, upper bound: 98.1271072
time: 138.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0250983, upper bound: 98.0677504
time: 122.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0029157, upper bound: 98.0695447
time: 415.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0634628, upper bound: 98.0137698
time: 148.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0137698, upper bound: 98.0634628
time: 340.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0695447, upper bound: 98.0029157
time: 492.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0677504, upper bound: 98.0250983
time: 138.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.1271072, upper bound: 97.9660982
time: 187.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 328.69 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 328.69
Output dim: 5, lower bound: -97.9660982, upper bound: 98.1271072
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 328.69
Output dim: 5, lower bound: -98.0250983, upper bound: 98.0677504
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 328.69
Output dim: 5, lower bound: -98.0029157, upper bound: 98.0695447
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 328.69
Output dim: 5, lower bound: -98.0634628, upper bound: 98.0137698
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 328.69
Output dim: 5, lower bound: -98.0137698, upper bound: 98.0634628
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 328.69
Output dim: 5, lower bound: -98.0695447, upper bound: 98.0029157
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 328.69
Output dim: 5, lower bound: -98.0677504, upper bound: 98.0250983
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 328.69
Output dim: 5, lower bound: -98.1271072, upper bound: 97.9660982

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9187122, upper bound: 98.1199552
time: 182.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9569244, upper bound: 98.0582834
time: 163.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9551221, upper bound: 98.0572639
time: 151.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9551221, upper bound: 98.0230562
time: 132.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9587226, upper bound: 98.0620506
time: 152.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9924232, upper bound: 97.9996356
time: 134.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9963195, upper bound: 98.0048038
time: 123.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0561928, upper bound: 97.9648337
time: 167.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.9648337, upper bound: 98.0561928
time: 187.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -98.0048038, upper bound: 97.9963195
time: 156.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9996356, upper bound: 97.9924232
time: 200.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0620505, upper bound: 97.9587226
time: 151.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -98.0230562, upper bound: 98.0181867
time: 317.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0572639, upper bound: 97.9551221
time: 163.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.0582834, upper bound: 97.9569244
time: 137.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -98.1199552, upper bound: 97.9187122
time: 173.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 313.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9187122, upper bound: 98.1199552
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9569244, upper bound: 98.0582834
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9551221, upper bound: 98.0572639
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9551221, upper bound: 98.0230562
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9587226, upper bound: 98.0620506
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9924232, upper bound: 97.9996356
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9963195, upper bound: 98.0048038
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -98.0561928, upper bound: 97.9648337
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9648337, upper bound: 98.0561928
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 313.56
Output dim: 5, lower bound: -98.0048038, upper bound: 97.9963195
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 313.56
Output dim: 5, lower bound: -97.9996356, upper bound: 97.9924232
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -98.0620505, upper bound: 97.9587226
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 313.56
Output dim: 5, lower bound: -98.0230562, upper bound: 98.0181867
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -98.0572639, upper bound: 97.9551221
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -98.0582834, upper bound: 97.9569244
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 313.56
Output dim: 5, lower bound: -98.1199552, upper bound: 97.9187122

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 647

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.8794522, upper bound: 98.0274678
time: 137.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -97.8548878, upper bound: 98.0789142
time: 200.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 647

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9154519, upper bound: 97.9693993
time: 242.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.8948044, upper bound: 98.0198693
time: 183.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -132.0919495, 88.9448853, -132.0919495, 88.9448853, -221.0368347, 221.0368347
1: -68.2628937, 68.5400696, -68.2628937, 68.5400696, -136.8029327, 136.8029480
2: -61.5559082, 70.9747009, -61.5559082, 70.9747009, -132.5306091, 132.5306091
3: -70.3478851, 83.5620270, -70.3478851, 83.5620270, -153.9098969, 153.9099121
4: -76.6335297, 82.3346863, -76.6335297, 82.3346863, -158.9682007, 158.9682159
5: -68.2535706, 85.8365555, -68.2535706, 85.8365555, -154.0901184, 154.0901184
6: -108.2601624, 80.8151703, -108.2601624, 80.8151703, -189.0753326, 189.0753174
7: -80.2512665, 81.0144348, -80.2512665, 81.0144348, -161.2656860, 161.2657013
8: -89.7248764, 102.2256927, -89.7248764, 102.2256927, -191.9505615, 191.9505615
9: -75.6580963, 80.5650940, -75.6580963, 80.5650940, -156.2231903, 156.2231750
10: -111.0808716, 105.0076065, -111.0808716, 105.0076065, -216.0884705, 216.0884705
11: -105.2509155, 64.8341446, -105.2509155, 64.8341446, -170.0850525, 170.0850525
12: -110.4729309, 86.0724182, -110.4729309, 86.0724182, -196.5453491, 196.5453491
13: -108.4209366, 109.7032089, -108.4209366, 109.7032089, -218.1241455, 218.1241455
14: -167.6108398, 95.4456635, -167.6108398, 95.4456635, -263.0565186, 263.0564575
15: -88.5074005, 76.3808136, -88.5074005, 76.3808136, -164.8882141, 164.8881989
16: -111.6021805, 80.3257446, -111.6021805, 80.3257446, -191.9279175, 191.9279022
17: -159.5187073, 84.1022263, -159.5187073, 84.1022263, -243.6209412, 243.6209412
18: -107.4778442, 79.4692001, -107.4778442, 79.4692001, -186.9470520, 186.9470520
19: -80.9901657, 48.7825508, -80.9901657, 48.7825508, -129.7727051, 129.7727203
20: -76.9305267, 60.8819199, -76.9305267, 60.8819199, -137.8124390, 137.8124390
21: -100.5399399, 60.4680061, -100.5399399, 60.4680061, -161.0079346, 161.0079346
22: -103.0187683, 62.2926674, -103.0187683, 62.2926674, -165.3114319, 165.3114319
23: -82.1662750, 61.7707062, -82.1662750, 61.7707062, -143.9369812, 143.9369812
24: -101.5549011, 63.1698875, -101.5549011, 63.1698875, -164.7247925, 164.7247620
25: -87.2504883, 66.2284622, -87.2504883, 66.2284622, -153.4789429, 153.4789429
26: -116.8764343, 94.5938416, -116.8764343, 94.5938416, -211.4702759, 211.4702759
27: -101.6215668, 66.8761292, -101.6215668, 66.8761292, -168.4976807, 168.4976959
28: -80.4167175, 67.9829407, -80.4167175, 67.9829407, -148.3996582, 148.3996429
29: -107.2915039, 58.5335503, -107.2915039, 58.5335503, -165.8250427, 165.8250427
30: -101.1416016, 74.4876251, -101.1416016, 74.4876251, -175.6291962, 175.6292114
31: -107.2774811, 67.2782288, -107.2774811, 67.2782288, -174.5557098, 174.5557098
32: -105.6582642, 69.6159592, -105.6582642, 69.6159592, -175.2742004, 175.2742004
33: -139.8687744, 92.3527222, -139.8687744, 92.3527222, -232.2214966, 232.2214966
34: -118.9467926, 64.1776962, -118.9467926, 64.1776962, -183.1244812, 183.1244812
35: -116.0291519, 71.7811661, -116.0291519, 71.7811661, -187.8103027, 187.8103027
36: -113.8380966, 71.7632141, -113.8380966, 71.7632141, -185.6013184, 185.6013031
37: -167.5968323, 74.4492111, -167.5968323, 74.4492111, -242.0460510, 242.0460510
38: -134.0997162, 83.4343262, -134.0997162, 83.4343262, -217.5340424, 217.5340424
39: -157.7188721, 87.7491074, -157.7188721, 87.7491074, -245.4679871, 245.4679871
40: -124.4302979, 72.0774841, -124.4302979, 72.0774841, -196.5077515, 196.5077667
41: -112.2657623, 81.1701050, -112.2657623, 81.1701050, -193.4358521, 193.4358673
42: -79.4125519, 72.6106339, -79.4125519, 72.6106339, -152.0231934, 152.0231781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=604, inp2_unstable=604, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=793, inp2_unstable=793, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=32, inp2_unstable=32, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 990
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 989
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 977
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 974
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 958
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 988
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 820

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 647

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.9167451, upper bound: 97.9647027
time: 146.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -97.8925979, upper bound: 98.0166003
time: 124.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 273.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 273.13
Output dim: 5, lower bound: -97.8794522, upper bound: 98.0274678
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 273.13
Output dim: 5, lower bound: -97.8548878, upper bound: 98.0789142
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 273.13
Output dim: 5, lower bound: -97.9154519, upper bound: 97.9693993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 273.13
Output dim: 5, lower bound: -97.8948044, upper bound: 98.0198693
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 273.13
Output dim: 5, lower bound: -97.9167451, upper bound: 97.9647027
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 273.13
Output dim: 5, lower bound: -97.8925979, upper bound: 98.0166003
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 273.13
Output dim: 5, lower bound: -97.9587226, upper bound: 98.0620506
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 273.13
Output dim: 5, lower bound: -98.0561928, upper bound: 97.9648337
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 273.13
Output dim: 5, lower bound: -97.9648337, upper bound: 98.0561928
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 273.13
Output dim: 5, lower bound: -98.0620505, upper bound: 97.9587226
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 273.13
Output dim: 5, lower bound: -98.0572639, upper bound: 97.9551221
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 273.13
Output dim: 5, lower bound: -98.0582834, upper bound: 97.9569244
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 273.13
Output dim: 5, lower bound: -98.1199552, upper bound: 97.9187122

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 203.95 + 7195.97 = 7399.92 seconds
